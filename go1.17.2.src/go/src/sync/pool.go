// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"internal/race"
	"runtime"
	"sync/atomic"
	"unsafe"
)

// A Pool is a set of temporary objects that may be individually saved and
// retrieved.
//
// Any item stored in the Pool may be removed automatically at any time without
// notification. If the Pool holds the only reference when this happens, the
// item might be deallocated.
//
// A Pool is safe for use by multiple goroutines simultaneously.
//
// Pool's purpose is to cache allocated but unused items for later reuse,
// relieving pressure on the garbage collector. That is, it makes it easy to
// build efficient, thread-safe free lists. However, it is not suitable for all
// free lists.
//
// An appropriate use of a Pool is to manage a group of temporary items
// silently shared among and potentially reused by concurrent independent
// clients of a package. Pool provides a way to amortize allocation overhead
// across many clients.
//
// An example of good use of a Pool is in the fmt package, which maintains a
// dynamically-sized store of temporary output buffers. The store scales under
// load (when many goroutines are actively printing) and shrinks when
// quiescent.
//
// On the other hand, a free list maintained as part of a short-lived object is
// not a suitable use for a Pool, since the overhead does not amortize well in
// that scenario. It is more efficient to have such objects implement their own
// free list.
//
// A Pool must not be copied after first use.
type Pool struct {
	noCopy noCopy

	// 为local数组，大小为P的个数，每个P一个local，使用p的id作为index
	local     unsafe.Pointer // local fixed-size per-P pool, actual type is [P]poolLocal
	// local数组的大小
	localSize uintptr        // size of the local array

	// 上一个周期的local
	victim     unsafe.Pointer // local from previous cycle
	// 上一个周期localSize
	victimSize uintptr        // size of victims array

	// New optionally specifies a function to generate
	// a value when Get would otherwise return nil.
	// It may not be changed concurrently with calls to Get.
	// pool不存在值时，初始化新值调用的函数
	New func() interface{}
}

// 1.默认先放在private里面
// 2.如果已经放了，则放到shared里面
// 3.shared是Local P能够push也能pop，其它P只能pop
// 4.Get的时候，先从private里面获取，没有，从shared里面获取，没有的话，从其它的poolChain获取
type poolLocalInternal struct {
	// 如果存储变量，用作Get和Put比较均衡的时候，直接从private里面获取，会更加快速
	private interface{} // Can be used only by the respective P.
	// private里已经存储了值，后续数据放入的地方
	shared  poolChain   // Local P can pushHead/popHead; any P can popTail.
}

// local的实现，使用链表
type poolLocal struct {
	poolLocalInternal

	// Prevents false sharing on widespread platforms with
	// 128 mod (cache line size) = 0 .
	pad [128 - unsafe.Sizeof(poolLocalInternal{})%128]byte
}

// from runtime
func fastrand() uint32

var poolRaceHash [128]uint64

// poolRaceAddr returns an address to use as the synchronization point
// for race detector logic. We don't use the actual pointer stored in x
// directly, for fear of conflicting with other synchronization on that address.
// Instead, we hash the pointer to get an index into poolRaceHash.
// See discussion on golang.org/cl/31589.
func poolRaceAddr(x interface{}) unsafe.Pointer {
	ptr := uintptr((*[2]unsafe.Pointer)(unsafe.Pointer(&x))[1])
	h := uint32((uint64(uint32(ptr)) * 0x85ebca6b) >> 16)
	return unsafe.Pointer(&poolRaceHash[h%uint32(len(poolRaceHash))])
}

// Put函数
// 将使用完的变量x归还回去，否则使用Get，会一直New新的变量出来
// 1.x不能为nil，否则直接返回
// 2.runtime_procPin到runtime_procUnpin期间，能保证当前goroutine不会被强占，保证数据操作安全
// 3.因为local使用P的id作为index，调用期间使用了runtime_procPin到runtime_procUnpin，
// 保证了对于local只有一个goroutine，从而保证了数据操作的安全
func (p *Pool) Put(x interface{}) {
	if x == nil {
		return
	}
	if race.Enabled {
		if fastrand()%4 == 0 {
			// Randomly drop x on floor.
			return
		}
		race.ReleaseMerge(poolRaceAddr(x))
		race.Disable()
	}
	l, _ := p.pin()
	if l.private == nil {
		l.private = x
		x = nil
	}
	if x != nil {
		l.shared.pushHead(x)
	}
	runtime_procUnpin()
	if race.Enabled {
		race.Enable()
	}
}

// 1.先从private获取
// 2.从shared里面获取
// 3.从其它的shared获取
func (p *Pool) Get() interface{} {
	if race.Enabled {
		race.Disable()
	}
	// 获取当前的P对应的local
	l, pid := p.pin()
	// 先从private获取，private不存在，则从
	x := l.private
	// 从private获取后，将private设置为nil
	l.private = nil
	if x == nil {
		// Try to pop the head of the local shard. We prefer
		// the head over the tail for temporal locality of
		// reuse.
		// private不存在，则从当前P的shared里面获取，获取第一个
		x, _ = l.shared.popHead()
		if x == nil {
			// shared不存在，则从其它的P的shared里面获取
			x = p.getSlow(pid)
		}
	}
	runtime_procUnpin()
	if race.Enabled {
		race.Enable()
		if x != nil {
			race.Acquire(poolRaceAddr(x))
		}
	}
	// 最后也未获取到，则直接New一个
	if x == nil && p.New != nil {
		x = p.New()
	}
	return x
}

// private、当前P的shared里面都没有的话，尝试从其它的P的shared里面获取
func (p *Pool) getSlow(pid int) interface{} {
	// See the comment in pin regarding ordering of the loads.
	size := runtime_LoadAcquintptr(&p.localSize) // load-acquire
	locals := p.local                            // load-consume
	// 获取locals数组，里面存贮了所有P的local
	// 尝试从其它P的share里面去偷一个元素出来
	for i := 0; i < int(size); i++ {
		l := indexLocal(locals, (pid+i+1)%int(size))
		if x, _ := l.shared.popTail(); x != nil {
			return x
		}
	}

	// 如果其它的share里面也没有，则尝试从victim里面获取，victim实则为执行完poolCleanup
	// 后，保存的上一个周期缓存的数据，会在下一个周期清除掉
	// 获取原理跟从local里获取一样
	size = atomic.LoadUintptr(&p.victimSize)
	if uintptr(pid) >= size {
		return nil
	}
	locals = p.victim
	l := indexLocal(locals, pid)
	if x := l.private; x != nil {
		l.private = nil
		return x
	}
	for i := 0; i < int(size); i++ {
		l := indexLocal(locals, (pid+i)%int(size))
		if x, _ := l.shared.popTail(); x != nil {
			return x
		}
	}

	// Mark the victim cache as empty for future gets don't bother
	// with it.
	atomic.StoreUintptr(&p.victimSize, 0)

	return nil
}

// 1.将当前goroutine绑定到P，禁止强占，并且返回属于当前P的localPool
// 2.必须与runtime_procUnpin搭配使用
// 3.该函数Put和Get均会调用
func (p *Pool) pin() (*poolLocal, int) {
	// 与runtime_procUnpin搭配使用
	pid := runtime_procPin()

	// 获取当前p的localSize，默认初始化，s为0，会进入pinSlow分支
	// 当前已经有调用Put操作，localSize已经不为0，才会进入
	s := runtime_LoadAcquintptr(&p.localSize) // load-acquire
	l := p.local                              // load-consume
	if uintptr(pid) < s {
		return indexLocal(l, pid), pid
	}
	// 通过pinSlow获取，命名规则跟Mutex的lockSlow类似
	return p.pinSlow()
}

// 1.初始化localPool会使用到全局锁allPoolsMu，因为所有的Pool都会装入allPools里面
// 2.先上一个全局锁，然后进行double check，检查local是否已经初始化，如果已经初始化，直接返回 如果为初始化，则进行初始化
// 3.使用P的id作为index，P的总个数作为local数组的大小
func (p *Pool) pinSlow() (*poolLocal, int) {
	// Retry under the mutex.
	// Can not lock the mutex while pinned.
	// 先Unpin当前P
	runtime_procUnpin()
	// 会使用全局锁allPoolsMu
	allPoolsMu.Lock()
	defer allPoolsMu.Unlock()
	pid := runtime_procPin()
	// poolCleanup won't be called while we are pinned.
	// poolCleanup清理函数，不会在pinned期间执行
	s := p.localSize
	l := p.local
	// double check，检查local是否已经初始化，如果已经初始化，则直接返回
	if uintptr(pid) < s {
		return indexLocal(l, pid), pid
	}
	// 将p加入allPools里面
	if p.local == nil {
		allPools = append(allPools, p)
	}
	// 如果GOMAXPROCS在GC阶段变化了，则重新赋值数组，并丢弃老的那个
	size := runtime.GOMAXPROCS(0)
	local := make([]poolLocal, size)
	// 初始化一个local，并将local的地址赋值给p.local
	atomic.StorePointer(&p.local, unsafe.Pointer(&local[0])) // store-release
	// 何止localSize的值，为当前P的个数
	runtime_StoreReluintptr(&p.localSize, uintptr(size))     // store-release
	//  返回第P的id个local的值和pid
	return &local[pid], pid
}

func poolCleanup() {
	// Drop victim caches from all pools.
	// 将老的victim设置为nil，这样gc便能清理
	for _, p := range oldPools {
		p.victim = nil
		p.victimSize = 0
	}

	// 将当前的pool里的local赋值给victim，并将local
	for _, p := range allPools {
		p.victim = p.local
		p.victimSize = p.localSize
		p.local = nil
		p.localSize = 0
	}

	// The pools with non-empty primary caches now have non-empty
	// victim caches and no pools have primary caches.
	// 将当前的pool赋值给oldPools，当前的pools清空
	oldPools, allPools = allPools, nil
}

var (
	allPoolsMu Mutex

	// allPools is the set of pools that have non-empty primary
	// caches. Protected by either 1) allPoolsMu and pinning or 2)
	// STW.
	allPools []*Pool

	// oldPools is the set of pools that may have non-empty victim
	// caches. Protected by STW.
	oldPools []*Pool
)

// 通过每次调用gc触发
func init() {
	// 注册poolCleanup到触发点
	runtime_registerPoolCleanup(poolCleanup)
}

func indexLocal(l unsafe.Pointer, i int) *poolLocal {
	lp := unsafe.Pointer(uintptr(l) + uintptr(i)*unsafe.Sizeof(poolLocal{}))
	return (*poolLocal)(lp)
}

// Implemented in runtime.
func runtime_registerPoolCleanup(cleanup func())
func runtime_procPin() int
func runtime_procUnpin()

// The below are implemented in runtime/internal/atomic and the
// compiler also knows to intrinsify the symbol we linkname into this
// package.

//go:linkname runtime_LoadAcquintptr runtime/internal/atomic.LoadAcquintptr
func runtime_LoadAcquintptr(ptr *uintptr) uintptr

//go:linkname runtime_StoreReluintptr runtime/internal/atomic.StoreReluintptr
func runtime_StoreReluintptr(ptr *uintptr, val uintptr) uintptr
