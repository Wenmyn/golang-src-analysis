// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"sync/atomic"
	"unsafe"
)

// Map is like a Go map[interface{}]interface{} but is safe for concurrent use
// by multiple goroutines without additional locking or coordination.
// Loads, stores, and deletes run in amortized constant time.
//
// The Map type is specialized. Most code should use a plain Go map instead,
// with separate locking or coordination, for better type safety and to make it
// easier to maintain other invariants along with the map content.
//
// The Map type is optimized for two common use cases: (1) when the entry for a given
// key is only ever written once but read many times, as in caches that only grow,
// or (2) when multiple goroutines read, write, and overwrite entries for disjoint
// sets of keys. In these two cases, use of a Map may significantly reduce lock
// contention compared to a Go map paired with a separate Mutex or RWMutex.
//
// The zero Map is empty and ready for use. A Map must not be copied after first use.
type Map struct {
	mu Mutex

	// 持有部分map的数据并且并发读不需要持有锁，但是改变指针需要持有锁
	// 注：无论是Load()还是Store()都需要先对read中的数据进行判断
	read atomic.Value // readOnly

	// 包含部分map数据需要持有锁保护 为了保证dirty map能够快速晋升为read map
	// 它还包含所有在read map未清除的数据
	// expunged数据不会存储在dirty map
	// 如果dirtymap为nil则下一次写会从read map拷贝一份数据
	dirty map[interface{}]*entry

	// 记录从read map中读不到数据，加锁去判断key是否存在的次数
	// 当misses等于dirty map长度时，dirty map会直接晋升为read map
	// 下次写操作再从read map拷贝一份数据
	misses int
}

// readOnly 是一个不可变结构体 自动存储在Map.read字段中
type readOnly struct {
	m       map[interface{}]*entry
	amended bool // 如果key在dirty不在m中 则为true如果为false则表示dirty为空
}

// expunged是一个任意指针，标记已删除的条目
// from the dirty map.
var expunged = unsafe.Pointer(new(interface{}))

// An entry is a slot in the map corresponding to a particular key.
type entry struct {
	// p指向interface{}类型的值
	// 如果p==nil，表明entry已经被删除并且m.dirty==nil
	// 如果p==expunged，表明entry已经被删除且m.dirty!=nil 并且entry不在m.dirty中
	// 否则entry是有效的value,并且记录m.read.m[key] 并且如果m.dirty!=nil 则也存在m.dirty[key]

	// 当m.dirty重建的时候被删除的entry会被检测到并自动由nil替换为expunged 并且不会设置m.dirty[key]
	// 若p!=expunged，则entry关联的值可以通过原子替换来更新
	// 若p==expunged，则需要先设置m.dirty[key]=e之后才能更新entry关联的值，这样以便使用dirty map查找值
	p unsafe.Pointer // *interface{}
}

func newEntry(i interface{}) *entry {
	return &entry{p: unsafe.Pointer(&i)}
}

// Load操作返回存储在map中指定key的value，有两个返回值，
// ok表示key对应的value是否存在。
func (m *Map) Load(key interface{}) (value interface{}, ok bool) {
	read, _ := m.read.Load().(readOnly)
	e, ok := read.m[key]
	// 不存在，且dirty中有新数据
	if !ok && read.amended {
		m.mu.Lock()	//加锁
		// 双检查，避免加锁的时候m.dirty提升为m.read,这个时候m.read可能被替换了。
		read, _ = m.read.Load().(readOnly)
		e, ok = read.m[key]
		//不存在 且map.dirty包含了部分不在read map中的数据
		if !ok && read.amended {
			e, ok = m.dirty[key]
			// 不管m.dirty中存不存在，都将misses计数加一
			// missLocked()中满足条件后就会提升m.dirty
			m.missLocked()
		}
		m.mu.Unlock()
	}
	if !ok {
		return nil, false
	}
	return e.load()	// 使用原子操作读取数据
}

func (e *entry) load() (value interface{}, ok bool) {
	p := atomic.LoadPointer(&e.p)
	if p == nil || p == expunged {
		return nil, false
	}
	return *(*interface{})(p), true
}

// 更新或者新增一个entry
func (m *Map) Store(key, value interface{}) {
	// 情况1：如果key已经在read map存在了 且dirty map中未被删除
	read, _ := m.read.Load().(readOnly)
	// 如果m.read存在这个键，并且这个entry没有被标记删除，尝试直接存储。
	// 因为m.dirty也指向这个entry,所以m.dirty也保持最新的entry。
	if e, ok := read.m[key]; ok && e.tryStore(&value) {
		return
	}

	m.mu.Lock()
	read, _ = m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok {
		if e.unexpungeLocked() {	//情况2：如果key存在于read map，不在dirty map中
			// The entry was previously expunged, which implies that there is a
			// non-nil dirty map and this entry is not in it.
			m.dirty[key] = e
		}
		e.storeLocked(&value)	// 情况3：如果key同时存在于read map和dirty map（mutex的影响）
	} else if e, ok := m.dirty[key]; ok {	//情况4：如果key不在read map但存在于dirty map
		e.storeLocked(&value)
	} else {	//情况5：如果key不在read map也不存在于dirty map
		if !read.amended {
			// 首先判断read.amended若为false
			// 则表明dirty map刚刚晋升为read map,此时dirty map为nil。然后调用m.dirtyLocked()
			m.dirtyLocked()
			m.read.Store(readOnly{m: read.m, amended: true})
		}
		// 将read map中的数据引用拷贝一份后，
		// 针对新来的数据 m.dirty[key] = newEntry(value)新建并插入
		m.dirty[key] = newEntry(value)
	}
	m.mu.Unlock()
}

// 如果key已经在read map存在了 且dirty map中未被删除
// 通过调用e.tryStore(&value)直接无锁情况下,更新对应值。前提是对应值 非expunged
func (e *entry) tryStore(i *interface{}) bool {
	for {
		p := atomic.LoadPointer(&e.p)
		if p == expunged {	//表示dirty map已删除对应值 返回false 让其更新完dirty map 再更新read map
			return false
		}
		if atomic.CompareAndSwapPointer(&e.p, p, unsafe.Pointer(i)) {	// 直接更新值
			return true
		}
	}
}

// unexpungeLocked ensures that the entry is not marked as expunged.
//
// If the entry was previously expunged, it must be added to the dirty map
// before m.mu is unlocked.
func (e *entry) unexpungeLocked() (wasExpunged bool) {
	return atomic.CompareAndSwapPointer(&e.p, expunged, nil)
}

// storeLocked unconditionally stores a value to the entry.
//
// The entry must be known not to be expunged.
func (e *entry) storeLocked(i *interface{}) {
	atomic.StorePointer(&e.p, unsafe.Pointer(i))
}

// LoadOrStore returns the existing value for the key if present.
// Otherwise, it stores and returns the given value.
// The loaded result is true if the value was loaded, false if stored.
func (m *Map) LoadOrStore(key, value interface{}) (actual interface{}, loaded bool) {
	// Avoid locking if it's a clean hit.
	read, _ := m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok {
		actual, loaded, ok := e.tryLoadOrStore(value)
		if ok {
			return actual, loaded
		}
	}

	m.mu.Lock()
	read, _ = m.read.Load().(readOnly)
	if e, ok := read.m[key]; ok {
		if e.unexpungeLocked() {
			m.dirty[key] = e
		}
		actual, loaded, _ = e.tryLoadOrStore(value)
	} else if e, ok := m.dirty[key]; ok {
		actual, loaded, _ = e.tryLoadOrStore(value)
		m.missLocked()
	} else {
		if !read.amended {
			// We're adding the first new key to the dirty map.
			// Make sure it is allocated and mark the read-only map as incomplete.
			m.dirtyLocked()
			m.read.Store(readOnly{m: read.m, amended: true})
		}
		m.dirty[key] = newEntry(value)
		actual, loaded = value, false
	}
	m.mu.Unlock()

	return actual, loaded
}

// tryLoadOrStore atomically loads or stores a value if the entry is not
// expunged.
//
// If the entry is expunged, tryLoadOrStore leaves the entry unchanged and
// returns with ok==false.
func (e *entry) tryLoadOrStore(i interface{}) (actual interface{}, loaded, ok bool) {
	p := atomic.LoadPointer(&e.p)
	if p == expunged {
		return nil, false, false
	}
	if p != nil {
		return *(*interface{})(p), true, true
	}

	// Copy the interface after the first load to make this method more amenable
	// to escape analysis: if we hit the "load" path or the entry is expunged, we
	// shouldn't bother heap-allocating.
	ic := i
	for {
		if atomic.CompareAndSwapPointer(&e.p, nil, unsafe.Pointer(&ic)) {
			return i, false, true
		}
		p = atomic.LoadPointer(&e.p)
		if p == expunged {	//删除直接返回
			return nil, false, false
		}
		if p != nil {	// 有值了 返回值
			return *(*interface{})(p), true, true
		}
	}
	// 注：如果entry不是expunged 则自动加载或存储值,
	// 如果entry为expunged 则直接返回 并且loaded和ok为false
}

// LoadAndDelete deletes the value for a key, returning the previous value if any.
// The loaded result reports whether the key was present.
func (m *Map) LoadAndDelete(key interface{}) (value interface{}, loaded bool) {
	read, _ := m.read.Load().(readOnly)
	e, ok := read.m[key]
	if !ok && read.amended {
		m.mu.Lock()
		read, _ = m.read.Load().(readOnly)
		e, ok = read.m[key]
		if !ok && read.amended {
			e, ok = m.dirty[key]
			delete(m.dirty, key)
			// Regardless of whether the entry was present, record a miss: this key
			// will take the slow path until the dirty map is promoted to the read
			// map.
			m.missLocked()
		}
		m.mu.Unlock()
	}
	if ok {
		return e.delete()
	}
	return nil, false
}

// Delete deletes the value for a key.
func (m *Map) Delete(key interface{}) {
	m.LoadAndDelete(key)
}

func (e *entry) delete() (value interface{}, ok bool) {
	for {
		p := atomic.LoadPointer(&e.p)
		if p == nil || p == expunged {
			return nil, false
		}
		if atomic.CompareAndSwapPointer(&e.p, p, nil) {
			return *(*interface{})(p), true
		}
	}
}

// Range calls f sequentially for each key and value present in the map.
// If f returns false, range stops the iteration.
//
// Range does not necessarily correspond to any consistent snapshot of the Map's
// contents: no key will be visited more than once, but if the value for any key
// is stored or deleted concurrently, Range may reflect any mapping for that key
// from any point during the Range call.
//
// Range may be O(N) with the number of elements in the map even if f returns
// false after a constant number of calls.
func (m *Map) Range(f func(key, value interface{}) bool) {
	// We need to be able to iterate over all of the keys that were already
	// present at the start of the call to Range.
	// If read.amended is false, then read.m satisfies that property without
	// requiring us to hold m.mu for a long time.
	read, _ := m.read.Load().(readOnly)
	if read.amended {
		// m.dirty contains keys not in read.m. Fortunately, Range is already O(N)
		// (assuming the caller does not break out early), so a call to Range
		// amortizes an entire copy of the map: we can promote the dirty copy
		// immediately!
		m.mu.Lock()
		read, _ = m.read.Load().(readOnly)
		if read.amended {
			read = readOnly{m: m.dirty}
			m.read.Store(read)
			m.dirty = nil
			m.misses = 0
		}
		m.mu.Unlock()
	}

	for k, e := range read.m {
		v, ok := e.load()
		if !ok {
			continue
		}
		if !f(k, v) {
			break
		}
	}
}

// Map.misses += 1, 如果misses == len(dirty) ,dirty升级为read ，然后dirty指向nil
func (m *Map) missLocked() {
	m.misses++
	if m.misses < len(m.dirty) {
		return
	}
	m.read.Store(readOnly{m: m.dirty})
	m.dirty = nil
	m.misses = 0
}

func (m *Map) dirtyLocked() {
	if m.dirty != nil {	//不为nil 直接返回
		return
	}

	read, _ := m.read.Load().(readOnly)
	m.dirty = make(map[interface{}]*entry, len(read.m))
	for k, e := range read.m {
		if !e.tryExpungeLocked() {	//删除则跳过复制
			m.dirty[k] = e	//从read map拷贝一份数据引用给到dirty map
		}
	}
}

// 判断read map中的数据是否为nil 若是 则将其更新为expunged
// 同时数据也不会copy到dirty map，所以expunged表明数据已经从dirty map移除了
func (e *entry) tryExpungeLocked() (isExpunged bool) {
	p := atomic.LoadPointer(&e.p)
	for p == nil {
		if atomic.CompareAndSwapPointer(&e.p, nil, expunged) {
			return true
		}
		p = atomic.LoadPointer(&e.p)
	}
	return p == expunged
}
