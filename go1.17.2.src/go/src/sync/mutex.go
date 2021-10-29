// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sync provides basic synchronization primitives such as mutual
// exclusion locks. Other than the Once and WaitGroup types, most are intended
// for use by low-level library routines. Higher-level synchronization is
// better done via channels and communication.
//
// Values containing the types defined in this package should not be copied.
package sync

import (
	"internal/race"
	"sync/atomic"
	"unsafe"
)

func throw(string) // provided by runtime

// A Mutex is a mutual exclusion lock.
// The zero value for a Mutex is an unlocked mutex.
//
// A Mutex must not be copied after first use.
type Mutex struct {
	state int32
	sema  uint32
}
// state字段表示当前互斥锁的状态信息，它是int32类型，其低三位的二进制位均有相应的状态含义。
// 1.mutexLocked是state中的低1位，用二进制表示为0001（为了方便，这里只描述后4位），它代表该互斥锁是否被加锁。
// 2.mutexWoken是低2位，用二进制表示为0010，它代表互斥锁上是否有被唤醒的goroutine。
// 3.mutexStarving是低3位，用二进制表示为0100，它代表当前互斥锁是否处于饥饿模式。

// 注：
// 1.state剩下的29位用于统计在互斥锁上的等待队列中goroutine数目（waiter）。
// 2.sema字段是信号量，用于控制goroutine的阻塞与唤醒，下文中会有介绍到。

// A Locker represents an object that can be locked and unlocked.
type Locker interface {
	Lock()
	Unlock()
}

const (
	mutexLocked = 1 << iota // mutex is locked
	mutexWoken
	mutexStarving
	mutexWaiterShift = iota	// mutexWaiterShift值为3，通过右移3位的位运算，可计算waiter个数

	// Mutex fairness.
	//
	// Mutex can be in 2 modes of operations: normal and starvation.
	// In normal mode waiters are queued in FIFO order, but a woken up waiter
	// does not own the mutex and competes with new arriving goroutines over
	// the ownership. New arriving goroutines have an advantage -- they are
	// already running on CPU and there can be lots of them, so a woken up
	// waiter has good chances of losing. In such case it is queued at front
	// of the wait queue. If a waiter fails to acquire the mutex for more than 1ms,
	// it switches mutex to the starvation mode.
	//
	// In starvation mode ownership of the mutex is directly handed off from
	// the unlocking goroutine to the waiter at the front of the queue.
	// New arriving goroutines don't try to acquire the mutex even if it appears
	// to be unlocked, and don't try to spin. Instead they queue themselves at
	// the tail of the wait queue.
	//
	// If a waiter receives ownership of the mutex and sees that either
	// (1) it is the last waiter in the queue, or (2) it waited for less than 1 ms,
	// it switches mutex back to normal operation mode.
	//
	// Normal mode has considerably better performance as a goroutine can acquire
	// a mutex several times in a row even if there are blocked waiters.
	// Starvation mode is important to prevent pathological cases of tail latency.
	starvationThresholdNs = 1e6	// 1ms，进入饥饿状态的等待时间
)

// Lock locks m.
// If the lock is already in use, the calling goroutine
// blocks until the mutex is available.
func (m *Mutex) Lock() {
	// 首先通过CAS判断当前锁的状态
	// 如果锁是完全空闲的，即m.state为0，则对其加锁，将m.state的值赋为1
	if atomic.CompareAndSwapInt32(&m.state, 0, mutexLocked) {
		if race.Enabled {
			race.Acquire(unsafe.Pointer(m))
		}
		return
	}
	// Slow path (outlined so that the fast path can be inlined)
	m.lockSlow()
}

func (m *Mutex) lockSlow() {
	var waitStartTime int64	  // 用于计算waiter的等待时间
	starving := false		  // 饥饿模式标志
	awoke := false			  // 唤醒标志
	iter := 0				  // 统计当前goroutine的自旋次数
	old := m.state			  // 保存当前锁的状态
	for {
		// 判断是否能进入自旋
		if old&(mutexLocked|mutexStarving) == mutexLocked && runtime_canSpin(iter) {
			// !awoke 判断当前goroutine是不是在唤醒状态
			// old&mutexWoken == 0 表示没有其他正在唤醒的goroutine
			// old>>mutexWaiterShift != 0 表示等待队列中有正在等待的goroutine
			if !awoke && old&mutexWoken == 0 && old>>mutexWaiterShift != 0 &&
				// 尝试将当前锁的低2位的Woken状态位设置为1，表示已被唤醒
				// 这是为了通知在解锁Unlock()中不要再唤醒其他的waiter了
				atomic.CompareAndSwapInt32(&m.state, old, old|mutexWoken) {
				awoke = true
			}
			runtime_doSpin()
			iter++
			old = m.state
			continue
		}
		// old是锁当前的状态，new是期望的状态，以期于在后面的CAS操作中更改锁的状态
		new := old

		// 如果当前锁不是饥饿模式，则将new的低1位的Locked状态位设置为1，表示加锁
		if old&mutexStarving == 0 {
			new |= mutexLocked
		}

		// 如果当前锁已被加锁或者处于饥饿模式，则将waiter数加1，
		// 表示当前goroutine将被作为waiter置于等待队列队尾
		if old&(mutexLocked|mutexStarving) != 0 {
			new += 1 << mutexWaiterShift
		}

		// 如果当前锁处于饥饿模式，并且已被加锁，则将低3位的Starving状态位设置为1，表示饥饿
		if starving && old&mutexLocked != 0 {
			new |= mutexStarving
		}

		// 当awoke为true，则表明当前goroutine在自旋逻辑中，成功修改锁的Woken状态位为1
		if awoke {
			if new&mutexWoken == 0 {
				throw("sync: inconsistent mutex state")
			}
			// 将唤醒标志位Woken置回为0
			// 因为在后续的逻辑中，当前goroutine 1.要么是拿到锁了，2.要么是被阻塞。

			// 如果是阻塞状态，那就需要等待其他释放锁的goroutine来唤醒。
			// 假如其他goroutine在unlock的时候发现Woken的位置不是0，则就不会去唤醒，那该goroutine就无法再醒来加锁。
			new &^= mutexWoken
		}
		// 尝试将锁的状态更新为期望状态
		if atomic.CompareAndSwapInt32(&m.state, old, new) {
			// 如果锁的原状态既不是被获取状态，也不是处于饥饿模式
			// 那就直接返回，表示当前goroutine已获取到锁
			// 注：这里的再次检查表示之前的Lock()是否已经UnLock()
			if old&(mutexLocked|mutexStarving) == 0 {
				break // locked the mutex with CAS
			}
			// 如果走到这里，那就证明当前goroutine没有获取到锁
			// 这里判断waitStartTime != 0就证明当前goroutine之前已经等待过了，则需要将其放置在等待队列队头
			queueLifo := waitStartTime != 0
			if waitStartTime == 0 {
				// 如果之前没有等待过，就以现在的时间来初始化设置
				waitStartTime = runtime_nanotime()
			}
			// 阻塞等待
			runtime_SemacquireMutex(&m.sema, queueLifo, 1)
			starving = starving || runtime_nanotime()-waitStartTime > starvationThresholdNs
			old = m.state
			if old&mutexStarving != 0 {
				// If this goroutine was woken and mutex is in starvation mode,
				// ownership was handed off to us but mutex is in somewhat
				// inconsistent state: mutexLocked is not set and we are still
				// accounted as waiter. Fix that.
				if old&(mutexLocked|mutexWoken) != 0 || old>>mutexWaiterShift == 0 {
					throw("sync: inconsistent mutex state")
				}
				delta := int32(mutexLocked - 1<<mutexWaiterShift)
				if !starving || old>>mutexWaiterShift == 1 {
					// Exit starvation mode.
					// Critical to do it here and consider wait time.
					// Starvation mode is so inefficient, that two goroutines
					// can go lock-step infinitely once they switch mutex
					// to starvation mode.
					delta -= mutexStarving
				}
				atomic.AddInt32(&m.state, delta)
				break
			}
			awoke = true
			iter = 0
		} else {
			old = m.state
		}
	}

	if race.Enabled {
		race.Acquire(unsafe.Pointer(m))
	}
}

// Unlock unlocks m.
// It is a run-time error if m is not locked on entry to Unlock.
//
// A locked Mutex is not associated with a particular goroutine.
// It is allowed for one goroutine to lock a Mutex and then
// arrange for another goroutine to unlock it.
func (m *Mutex) Unlock() {
	if race.Enabled {
		_ = m.state
		race.Release(unsafe.Pointer(m))
	}

	// Fast path: drop lock bit.
	new := atomic.AddInt32(&m.state, -mutexLocked)
	if new != 0 {
		// Outlined slow path to allow inlining the fast path.
		// To hide unlockSlow during tracing we skip one extra frame when tracing GoUnblock.
		m.unlockSlow(new)
	}
}

func (m *Mutex) unlockSlow(new int32) {
	if (new+mutexLocked)&mutexLocked == 0 {
		throw("sync: unlock of unlocked mutex")
	}
	if new&mutexStarving == 0 {
		old := new
		for {
			// If there are no waiters or a goroutine has already
			// been woken or grabbed the lock, no need to wake anyone.
			// In starvation mode ownership is directly handed off from unlocking
			// goroutine to the next waiter. We are not part of this chain,
			// since we did not observe mutexStarving when we unlocked the mutex above.
			// So get off the way.
			if old>>mutexWaiterShift == 0 || old&(mutexLocked|mutexWoken|mutexStarving) != 0 {
				return
			}
			// Grab the right to wake someone.
			new = (old - 1<<mutexWaiterShift) | mutexWoken
			if atomic.CompareAndSwapInt32(&m.state, old, new) {
				runtime_Semrelease(&m.sema, false, 1)
				return
			}
			old = m.state
		}
	} else {
		// Starving mode: handoff mutex ownership to the next waiter, and yield
		// our time slice so that the next waiter can start to run immediately.
		// Note: mutexLocked is not set, the waiter will set it after wakeup.
		// But mutex is still considered locked if mutexStarving is set,
		// so new coming goroutines won't acquire it.
		runtime_Semrelease(&m.sema, true, 1)
	}
}
