// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"internal/race"
	"sync/atomic"
	"unsafe"
)

// If you make any changes here, see // There is a modified copy of this file in runtime/rwmutex.go.if you should make them there.

// A RWMutex is a reader/writer mutual exclusion lock.
// The lock can be held by an arbitrary number of readers or a single writer.
// The zero value for a RWMutex is an unlocked mutex.
//
// A RWMutex must not be copied after first use.
//
// If a goroutine holds a RWMutex for reading and another goroutine might
// call Lock, no goroutine should expect to be able to acquire a read lock
// until the initial read lock is released. In particular, this prohibits
// recursive read locking. This is to ensure that the lock eventually becomes
// available; a blocked Lock call excludes new readers from acquiring the
// lock.
type RWMutex struct {
	w           Mutex  // w 互斥锁
	writerSem   uint32 // writers信号量
	readerSem   uint32 // readers 信号量
	readerCount int32  // 进行中的读协程数量
	readerWait  int32  // 离开的读协程数量
}

const rwmutexMaxReaders = 1 << 30

// 1.加读锁很简单，就是将readerCount数量+1，
// 如果readerCount<0 就阻塞等待信号量释放，这个readerCount什么时候会小于0呢？
// 2.在加写锁时atomic.AddInt32(&rw.readerCount, -rwmutexMaxReaders)，rw.readerCount将会被减为负，
// 说明此时有写锁到来，读锁就应该阻塞在这里等待释放
func (rw *RWMutex) RLock() {
	if race.Enabled {
		_ = rw.w.state
		race.Disable()
	}
	if atomic.AddInt32(&rw.readerCount, 1) < 0 {
		// A writer is pending, wait for it.
		runtime_SemacquireMutex(&rw.readerSem, false, 0)
	}
	if race.Enabled {
		race.Enable()
		race.Acquire(unsafe.Pointer(&rw.readerSem))
	}
}

// 1.读锁释放，就是将rw.readerCount读协程数量-1，如果<0，说明有writer加了写锁，走慢解锁逻辑
func (rw *RWMutex) RUnlock() {
	if race.Enabled {
		_ = rw.w.state
		race.ReleaseMerge(unsafe.Pointer(&rw.writerSem))
		race.Disable()
	}
	//将readerCount -1 ，代表解读锁，如果小于0，说明有写锁在准备，走慢解锁
	if r := atomic.AddInt32(&rw.readerCount, -1); r < 0 {
		// Outlined slow-path to allow the fast-path to be inlined
		rw.rUnlockSlow(r)
	}
	if race.Enabled {
		race.Enable()
	}
}

// 1.读协程数量为1了，释放写信号量，上面阻塞的writer得到了继续执行
func (rw *RWMutex) rUnlockSlow(r int32) {
	if r+1 == 0 || r+1 == -rwmutexMaxReaders {
		race.Enable()
		throw("sync: RUnlock of unlocked RWMutex")
	}
	// 有写锁在准备，并且atomic.AddInt32(&rw.readerWait, -1) == 0，
	// 说明这是最后一个reader了，唤醒阻塞在上面的writer
	if atomic.AddInt32(&rw.readerWait, -1) == 0 {
		// The last reader unblocks the writer.
		runtime_Semrelease(&rw.writerSem, false, 1)
	}
}

// 1.加互斥锁，保证只有一个writer执行
// 2.重点在这句r := atomic.AddInt32(&rw.readerCount, -rwmutexMaxReaders) + rwmutexMaxReaders，
// 将readerCount置为负，加上一个最大数，最后又加回来，此时 r为readerCount的数量。
// ——减为负的作用：标识这里有个正在准备的writer
// 3.判断是不是r！=0,如果r！=0 将 rw.readerWait原子加上 +r，
// 代表需要离开的reader数量，最后阻塞等待写信号量释放
// 4.为什么要将readerCount赋值给readerWait了，因为后面会有很多reader加读锁，
// 执行完了就可以唤醒阻塞的writer了。readerCount已经不是原来的数量了，我们等重新赋值给新字段记录下才行。
func (rw *RWMutex) Lock() {
	if race.Enabled {
		_ = rw.w.state
		race.Disable()
	}
	//加互斥锁，写读占，当要写的时候，是独占，根互斥锁是一个原理
	rw.w.Lock()
	// Announce to readers there is a pending writer.
	r := atomic.AddInt32(&rw.readerCount, -rwmutexMaxReaders) + rwmutexMaxReaders
	// Wait for active readers.
	if r != 0 && atomic.AddInt32(&rw.readerWait, r) != 0 {
		runtime_SemacquireMutex(&rw.writerSem, false, 0)
	}
	if race.Enabled {
		race.Enable()
		race.Acquire(unsafe.Pointer(&rw.readerSem))
		race.Acquire(unsafe.Pointer(&rw.writerSem))
	}
}


func (rw *RWMutex) Unlock() {
	if race.Enabled {
		_ = rw.w.state
		race.Release(unsafe.Pointer(&rw.readerSem))
		race.Disable()
	}

	// 将读数量加上最大数恢复reader数量
	r := atomic.AddInt32(&rw.readerCount, rwmutexMaxReaders)
	if r >= rwmutexMaxReaders {
		race.Enable()
		throw("sync: Unlock of unlocked RWMutex")
	}
	// 循环释放阻塞在上面的读协程，释放次数与writer后面的新reader数量相等
	for i := 0; i < int(r); i++ {
		runtime_Semrelease(&rw.readerSem, false, 0)
	}
	// Allow other writers to proceed.
	rw.w.Unlock()	//解锁互斥锁
	if race.Enabled {
		race.Enable()
	}
}

// 对Locker接口实现的一个封装
func (rw *RWMutex) RLocker() Locker {
	return (*rlocker)(rw)
}

type rlocker RWMutex

func (r *rlocker) Lock()   { (*RWMutex)(r).RLock() }
func (r *rlocker) Unlock() { (*RWMutex)(r).RUnlock() }
