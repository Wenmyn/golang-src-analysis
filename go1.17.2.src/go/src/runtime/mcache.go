// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"runtime/internal/atomic"
	"unsafe"
)

// Per-thread (in Go, per-P) cache for small objects.
// This includes a small object cache and local allocation stats.
// No locking needed because it is per-thread (per-P).
//
// mcaches are allocated from non-GC'd memory, so any heap pointers
// must be specially handled.
//
//go:notinheap
type mcache struct {
	// The following members are accessed on every malloc,
	// so they are grouped here for better caching.
	nextSample uintptr // trigger heap sample after allocating this many bytes
	scanAlloc  uintptr // bytes of scannable heap allocated

	// Allocator cache for tiny objects w/o pointers.
	// See "Tiny allocator" comment in malloc.go.

	// tiny points to the beginning of the current tiny block, or
	// nil if there is no current tiny block.
	//
	// tiny is a heap pointer. Since mcache is in non-GC'd memory,
	// we handle it by clearing it in releaseAll during mark
	// termination.
	//
	// tinyAllocs is the number of tiny allocations performed
	// by the P that owns this mcache.
	tiny       uintptr	// 分配器起始地址
	tinyoffset uintptr	// 下一个空闲地偏置
	tinyAllocs uintptr	// 已经分配的对象个数

	// The rest is not accessed on every malloc.

	alloc [numSpanClasses]*mspan // 申请的136个span
	// 栈链表
	stackcache [_NumStackOrders]stackfreelist

	// Local allocator stats, flushed during GC.
	flushGen uint32
}

// A gclink is a node in a linked list of blocks, like mlink,
// but it is opaque to the garbage collector.
// The GC does not trace the pointers during collection,
// and the compiler does not emit write barriers for assignments
// of gclinkptr values. Code should store references to gclinks
// as gclinkptr, not as *gclink.
type gclink struct {
	next gclinkptr
}

// A gclinkptr is a pointer to a gclink, but it is opaque
// to the garbage collector.
type gclinkptr uintptr

// ptr returns the *gclink form of p.
// The result should be used for accessing fields, not stored
// in other data structures.
func (p gclinkptr) ptr() *gclink {
	return (*gclink)(unsafe.Pointer(p))
}

type stackfreelist struct {
	list gclinkptr // linked list of free stacks
	size uintptr   // total size of stacks in list
}

// dummy mspan that contains no free objects.
var emptymspan mspan

func allocmcache() *mcache {
	var c *mcache
	systemstack(func() {
		lock(&mheap_.lock)
		c = (*mcache)(mheap_.cachealloc.alloc())
		c.flushGen = mheap_.sweepgen
		unlock(&mheap_.lock)
	})
	for i := range c.alloc {
		c.alloc[i] = &emptymspan
	}
	c.nextSample = nextSample()
	return c
}

// freemcache releases resources associated with this
// mcache and puts the object onto a free list.
//
// In some cases there is no way to simply release
// resources, such as statistics, so donate them to
// a different mcache (the recipient).
func freemcache(c *mcache) {
	systemstack(func() {
		c.releaseAll()
		stackcache_clear(c)

		// NOTE(rsc,rlh): If gcworkbuffree comes back, we need to coordinate
		// with the stealing of gcworkbufs during garbage collection to avoid
		// a race where the workbuf is double-freed.
		// gcworkbuffree(c.gcworkbuf)

		lock(&mheap_.lock)
		mheap_.cachealloc.free(unsafe.Pointer(c))
		unlock(&mheap_.lock)
	})
}

// getMCache is a convenience function which tries to obtain an mcache.
//
// Returns nil if we're not bootstrapping or we don't have a P. The caller's
// P must not change, so we must be in a non-preemptible state.
func getMCache() *mcache {
	// Grab the mcache, since that's where stats live.
	pp := getg().m.p.ptr()
	var c *mcache
	if pp == nil {
		// We will be called without a P while bootstrapping,
		// in which case we use mcache0, which is set in mallocinit.
		// mcache0 is cleared when bootstrapping is complete,
		// by procresize.
		c = mcache0
	} else {
		c = pp.mcache
	}
	return c
}

// refill acquires a new span of span class spc for c. This span will
// have at least one free object. The current span in c must be full.
//
// Must run in a non-preemptible context since otherwise the owner of
// c could change.
func (c *mcache) refill(spc spanClass) {
	// Return the current cached span to the central lists.
	s := c.alloc[spc]

	if uintptr(s.allocCount) != s.nelems {
		throw("refill of span with free space remaining")
	}
	if s != &emptymspan {
		// Mark this span as no longer cached.
		if s.sweepgen != mheap_.sweepgen+3 {
			throw("bad sweepgen in refill")
		}
		mheap_.central[spc].mcentral.uncacheSpan(s)
	}

	// 向central进行申请
	// cacheSpan是向mcentral申请span的核心接口
	s = mheap_.central[spc].mcentral.cacheSpan() // 下面会讲解
	if s == nil {
		throw("out of memory")
	}

	if uintptr(s.allocCount) == s.nelems {
		throw("span has no free space")
	}

	// Indicate that this span is cached and prevent asynchronous
	// sweeping in the next sweep phase.
	s.sweepgen = mheap_.sweepgen + 3

	// Assume all objects from this span will be allocated in the
	// mcache. If it gets uncached, we'll adjust this.
	stats := memstats.heapStats.acquire()
	atomic.Xadduintptr(&stats.smallAllocCount[spc.sizeclass()], uintptr(s.nelems)-uintptr(s.allocCount))

	// Flush tinyAllocs.
	if spc == tinySpanClass {
		atomic.Xadduintptr(&stats.tinyAllocCount, c.tinyAllocs)
		c.tinyAllocs = 0
	}
	memstats.heapStats.release()

	// Update gcController.heapLive with the same assumption.
	usedBytes := uintptr(s.allocCount) * s.elemsize
	atomic.Xadd64(&gcController.heapLive, int64(s.npages*pageSize)-int64(usedBytes))

	// While we're here, flush scanAlloc, since we have to call
	// revise anyway.
	atomic.Xadd64(&gcController.heapScan, int64(c.scanAlloc))
	c.scanAlloc = 0

	if trace.enabled {
		// gcController.heapLive changed.
		traceHeapAlloc()
	}
	if gcBlackenEnabled != 0 {
		// gcController.heapLive and heapScan changed.
		gcController.revise()
	}

	c.alloc[spc] = s
}

// 大对象申请
func (c *mcache) allocLarge(size uintptr, needzero bool, noscan bool) (*mspan, bool) {
	// 内存溢出判断
	if size+_PageSize < size {
		throw("out of memory")
	}
	// 计算出对象所需的页数 对于大于32K的大内存分配都是整数页
	npages := size >> _PageShift
	if size&_PageMask != 0 {
		npages++
	}

	deductSweepCredit(npages*_PageSize, npages)

	// 分配函数的具体实现，使用span->sizeclass=0
	spc := makeSpanClass(0, noscan)
	s, isZeroed := mheap_.alloc(npages, spc, needzero)	 // 下面会讲解
	if s == nil {
		throw("out of memory")
	}
	stats := memstats.heapStats.acquire()
	atomic.Xadduintptr(&stats.largeAlloc, npages*pageSize)
	atomic.Xadduintptr(&stats.largeAllocCount, 1)
	memstats.heapStats.release()

	// Update gcController.heapLive and revise pacing if needed.
	atomic.Xadd64(&gcController.heapLive, int64(npages*pageSize))
	if trace.enabled {
		// Trace that a heap alloc occurred because gcController.heapLive changed.
		traceHeapAlloc()
	}
	if gcBlackenEnabled != 0 {
		gcController.revise()
	}

	// Put the large span in the mcentral swept list so that it's
	// visible to the background sweeper.
	mheap_.central[spc].mcentral.fullSwept(mheap_.sweepgen).push(s)
	s.limit = s.base() + size
	heapBitsForAddr(s.base()).initSpan(s)
	return s, isZeroed
}

func (c *mcache) releaseAll() {
	// Take this opportunity to flush scanAlloc.
	atomic.Xadd64(&gcController.heapScan, int64(c.scanAlloc))
	c.scanAlloc = 0

	sg := mheap_.sweepgen
	for i := range c.alloc {
		s := c.alloc[i]
		if s != &emptymspan {
			// Adjust nsmallalloc in case the span wasn't fully allocated.
			n := uintptr(s.nelems) - uintptr(s.allocCount)
			stats := memstats.heapStats.acquire()
			atomic.Xadduintptr(&stats.smallAllocCount[spanClass(i).sizeclass()], -n)
			memstats.heapStats.release()
			if s.sweepgen != sg+1 {
				// refill conservatively counted unallocated slots in gcController.heapLive.
				// Undo this.
				//
				// If this span was cached before sweep, then
				// gcController.heapLive was totally recomputed since
				// caching this span, so we don't do this for
				// stale spans.
				atomic.Xadd64(&gcController.heapLive, -int64(n)*int64(s.elemsize))
			}
			// Release the span to the mcentral.
			mheap_.central[i].mcentral.uncacheSpan(s)
			c.alloc[i] = &emptymspan
		}
	}
	// Clear tinyalloc pool.
	c.tiny = 0
	c.tinyoffset = 0

	// Flush tinyAllocs.
	stats := memstats.heapStats.acquire()
	atomic.Xadduintptr(&stats.tinyAllocCount, c.tinyAllocs)
	c.tinyAllocs = 0
	memstats.heapStats.release()

	// Updated heapScan and possible gcController.heapLive.
	if gcBlackenEnabled != 0 {
		gcController.revise()
	}
}

// prepareForSweep flushes c if the system has entered a new sweep phase
// since c was populated. This must happen between the sweep phase
// starting and the first allocation from c.
func (c *mcache) prepareForSweep() {
	// Alternatively, instead of making sure we do this on every P
	// between starting the world and allocating on that P, we
	// could leave allocate-black on, allow allocation to continue
	// as usual, use a ragged barrier at the beginning of sweep to
	// ensure all cached spans are swept, and then disable
	// allocate-black. However, with this approach it's difficult
	// to avoid spilling mark bits into the *next* GC cycle.
	sg := mheap_.sweepgen
	if c.flushGen == sg {
		return
	} else if c.flushGen != sg-2 {
		println("bad flushGen", c.flushGen, "in prepareForSweep; sweepgen", sg)
		throw("bad flushGen")
	}
	c.releaseAll()
	stackcache_clear(c)
	atomic.Store(&c.flushGen, mheap_.sweepgen) // Synchronizes with gcStart
}
