// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// This file contains the implementation of Go's map type.
//
// A map is just a hash table. The data is arranged
// into an array of buckets. Each bucket contains up to
// 8 key/elem pairs. The low-order bits of the hash are
// used to select a bucket. Each bucket contains a few
// high-order bits of each hash to distinguish the entries
// within a single bucket.
//
// If more than 8 keys hash to a bucket, we chain on
// extra buckets.
//
// When the hashtable grows, we allocate a new array
// of buckets twice as big. Buckets are incrementally
// copied from the old bucket array to the new bucket array.
//
// Map iterators walk through the array of buckets and
// return the keys in walk order (bucket #, then overflow
// chain order, then bucket index).  To maintain iteration
// semantics, we never move keys within their bucket (if
// we did, keys might be returned 0 or 2 times).  When
// growing the table, iterators remain iterating through the
// old table and must check the new table if the bucket
// they are iterating through has been moved ("evacuated")
// to the new table.

// Picking loadFactor: too large and we have lots of overflow
// buckets, too small and we waste a lot of space. I wrote
// a simple program to check some stats for different loads:
// (64-bit, 8 byte keys and elems)
//  loadFactor    %overflow  bytes/entry     hitprobe    missprobe
//        4.00         2.13        20.77         3.00         4.00
//        4.50         4.05        17.30         3.25         4.50
//        5.00         6.85        14.77         3.50         5.00
//        5.50        10.55        12.94         3.75         5.50
//        6.00        15.27        11.67         4.00         6.00
//        6.50        20.90        10.79         4.25         6.50
//        7.00        27.14        10.15         4.50         7.00
//        7.50        34.03         9.73         4.75         7.50
//        8.00        41.10         9.40         5.00         8.00
//
// %overflow   = percentage of buckets which have an overflow bucket
// bytes/entry = overhead bytes used per key/elem pair
// hitprobe    = # of entries to check when looking up a present key
// missprobe   = # of entries to check when looking up an absent key
//
// Keep in mind this data is for maximally loaded tables, i.e. just
// before the table grows. Typical tables will be somewhat less loaded.

import (
	"runtime/internal/atomic"
	"runtime/internal/math"
	"runtime/internal/sys"
	"unsafe"
)

const (
	// 一个桶中最多能装载的键值对（key-value）的个数为8
	bucketCntBits = 3
	bucketCnt     = 1 << bucketCntBits

	// 触发扩容的装载因子为13/2=6.5
	loadFactorNum = 13
	loadFactorDen = 2

	// 键和值超过128个字节，就会被转换为指针
	maxKeySize  = 128
	maxElemSize = 128

	// 数据偏移量应该是bmap结构体的大小，它需要正确地对齐。
	// 对于amd64p32而言，这意味着：即使指针是32位的，也是64位对齐。
	dataOffset = unsafe.Offsetof(struct {
		b bmap
		v int64
	}{}.v)

	// 每个桶（如果有溢出，则包含它的overflow的链接桶）在搬迁完成状态（evacuated* states）下，
	// 要么会包含它所有的键值对，要么一个都不包含（但不包括调用evacuate()方法阶段，
	// 该方法调用只会在对map发起write时发生，在该阶段其他goroutine是无法查看该map的）。简单的说，桶里的数据要么一起搬走，要么一个都还未搬。
	emptyRest      = 0 // 表示cell为空，并且比它高索引位的cell或者overflows中的cell都是空的。（初始化bucket时，就是该状态）
	emptyOne       = 1 // 空的cell，cell已经被搬迁到新的bucket
	evacuatedX     = 2 // 键值对已经搬迁完毕，key在新buckets数组的前半部分
	evacuatedY     = 3 // 键值对已经搬迁完毕，key在新buckets数组的后半部分
	evacuatedEmpty = 4 // cell为空，整个bucket已经搬迁完毕
	minTopHash     = 5 // tophash的最小正常值

	// flags
	iterator     = 1 // 可能有迭代器在使用buckets
	oldIterator  = 2 // 可能有迭代器在使用oldbuckets
	hashWriting  = 4 // 有协程正在向map写人key
	sameSizeGrow = 8 // 等量扩容

	// 用于迭代器检查的bucket ID
	noCheck = 1<<(8*sys.PtrSize) - 1
)

// isEmpty reports whether the given tophash array entry represents an empty bucket entry.
func isEmpty(x uint8) bool {
	return x <= emptyOne
}

// A header for a Go map.
type hmap struct {
	count     int 		// 代表哈希表中的元素个数，调用len(map)时，返回的就是该字段值。
	flags     uint8 	// 状态标志，下文常量中会解释四种状态位含义。
	B         uint8 	// buckets（桶）的对数log_2（哈希表元素数量最大可达到装载因子*2^B）
	noverflow uint16 	// 溢出桶的大概数量。
	hash0     uint32	// 哈希种子。

	buckets    unsafe.Pointer // 指向buckets数组的指针，数组大小为2^B，如果元素个数为0，它为nil。
	oldbuckets unsafe.Pointer // 如果发生扩容，oldbuckets是指向老的buckets数组的指针，老的buckets数组大小是新的buckets的1/2。非扩容状态下，它为nil。
	nevacuate  uintptr        // 表示扩容进度，小于此地址的buckets代表已搬迁完成。

	extra *mapextra // 这个字段是为了优化GC扫描而设计的。当key和value均不包含指针，并且都可以inline时使用。extra是指向mapextra类型的指针。
}
// 注：1.B的数量根据元素数量呈现2的倍数增长，并不是要恰好选择能容纳N个元素的数量，目的是让元素分散。

// mapextra holds fields that are not present on all maps.
type mapextra struct {
	// 如果 key 和 value 都不包含指针，并且可以被 inline(<=128 字节)
	// 就使用 hmap的extra字段 来存储 overflow buckets，这样可以避免 GC 扫描整个 map
	// 然而 bmap.overflow 也是个指针。这时候我们只能把这些 overflow 的指针
	// 都放在 hmap.extra.overflow 和 hmap.extra.oldoverflow 中了
	overflow    *[]*bmap
	oldoverflow *[]*bmap

	// 指向空闲的 overflow bucket 的指针
	nextOverflow *bmap
}

// A bucket for a Go map.
type bmap struct {
	// tophash包含此桶中每个键的哈希值最高字节（高8位）信息（也就是前面所述的high-order bits）。
	// 如果tophash[0] < minTopHash，tophash[0]则代表桶的搬迁（evacuation）状态。
	tophash [bucketCnt]uint8
}

// A hash iteration structure.
// If you modify hiter, also change cmd/compile/internal/reflectdata/reflect.go to indicate
// the layout of this structure.

// map遍历时用到的结构，startBucket+offset设定了开始遍历的地址，保证map遍历的无序性
type hiter struct {
	// key的指针
	key         unsafe.Pointer // Must be in first position.  Write nil to indicate iteration end (see cmd/compile/internal/walk/range.go).
	// 当前value的指针
	elem        unsafe.Pointer // Must be in second position (see cmd/compile/internal/walk/range.go).
	t           *maptype
	// 指向map的指针
	h           *hmap
	// 指向buckets的指针
	buckets     unsafe.Pointer // bucket ptr at hash_iter initialization time
	// 指向当前遍历的bucket的指针
	bptr        *bmap          // current bucket
	// 指向map.extra.overflow
	overflow    *[]*bmap       // keeps overflow buckets of hmap.buckets alive
	// 指向map.extra.oldoverflow
	oldoverflow *[]*bmap       // keeps overflow buckets of hmap.oldbuckets alive
	// 开始遍历的bucket的索引
	startBucket uintptr        // bucket iteration started at
	// 开始遍历bucket上的偏移量
	offset      uint8          // intra-bucket offset to start from during iteration (should be big enough to hold bucketCnt-1)
	// 是否从头遍历了
	wrapped     bool           // already wrapped around from end of bucket array to beginning
	// B 的大小
	B           uint8
	// 指示当前 cell 序号
	i           uint8
	// 指向当前的 bucket
	bucket      uintptr
	// 因为扩容，需要检查的 bucket
	checkBucket uintptr
}

// bucketShift returns 1<<b, optimized for code generation.
func bucketShift(b uint8) uintptr {
	// Masking the shift amount allows overflow checks to be elided.
	return uintptr(1) << (b & (sys.PtrSize*8 - 1))
}

// bucketMask returns 1<<b - 1, optimized for code generation.
func bucketMask(b uint8) uintptr {
	return bucketShift(b) - 1
}

// 再用哈希值的高8位，找到此key在桶中的位置。
func tophash(hash uintptr) uint8 {
	top := uint8(hash >> (sys.PtrSize*8 - 8))
	if top < minTopHash {
		top += minTopHash
	}
	return top
}

func evacuated(b *bmap) bool {
	h := b.tophash[0]
	return h > emptyOne && h < minTopHash
}

func (b *bmap) overflow(t *maptype) *bmap {
	return *(**bmap)(add(unsafe.Pointer(b), uintptr(t.bucketsize)-sys.PtrSize))
}

func (b *bmap) setoverflow(t *maptype, ovf *bmap) {
	*(**bmap)(add(unsafe.Pointer(b), uintptr(t.bucketsize)-sys.PtrSize)) = ovf
}

func (b *bmap) keys() unsafe.Pointer {
	return add(unsafe.Pointer(b), dataOffset)
}

// incrnoverflow increments h.noverflow.
// noverflow counts the number of overflow buckets.
// This is used to trigger same-size map growth.
// See also tooManyOverflowBuckets.
// To keep hmap small, noverflow is a uint16.
// When there are few buckets, noverflow is an exact count.
// When there are many buckets, noverflow is an approximate count.
func (h *hmap) incrnoverflow() {
	// We trigger same-size map growth if there are
	// as many overflow buckets as buckets.
	// We need to be able to count to 1<<h.B.
	if h.B < 16 {
		h.noverflow++
		return
	}
	// Increment with probability 1/(1<<(h.B-15)).
	// When we reach 1<<15 - 1, we will have approximately
	// as many overflow buckets as buckets.
	mask := uint32(1)<<(h.B-15) - 1
	// Example: if h.B == 18, then mask == 7,
	// and fastrand & 7 == 0 with probability 1/8.
	if fastrand()&mask == 0 {
		h.noverflow++
	}
}

func (h *hmap) newoverflow(t *maptype, b *bmap) *bmap {
	var ovf *bmap
	if h.extra != nil && h.extra.nextOverflow != nil {
		// 如果在预先创建了 overflow 区域，则先从本区域进行获取
		ovf = h.extra.nextOverflow
		if ovf.overflow(t) == nil {
			// overflow() 是读取的该 bucket 最后一个指针空间，是否有值
			// 有则代表预先申请的 overflow 区域已经用完，还记得 makeBucketArray 最后的设置吗？
			// 是保存的 h.buckets 的起始地址
			// 然后 nextOverFlow 维护预申请 overflow 域内的偏移量
			h.extra.nextOverflow = (*bmap)(add(unsafe.Pointer(ovf), uintptr(t.bucketsize)))
		} else {
			// 预先的已经用完了。此时的 ovf 代表了 overflow 内最后一个 bucket，将最后的指针位设置为 空
			// 并标记下预先申请的 overflow 区域已经用完
			ovf.setoverflow(t, nil)
			h.extra.nextOverflow = nil
		}
	} else {
		ovf = (*bmap)(newobject(t.bucket))
	}
	h.incrnoverflow()
	// 注：是否使用hmap.extra.overflow字段，取决于是否_maptype.bucket.ptrdata==0
	if t.bucket.ptrdata == 0 {
		h.createOverflow()
		*h.extra.overflow = append(*h.extra.overflow, ovf)
	}
	// setoverflow 就是将一个节点添加到某个节点的后方，一般就是末位节点（链表结构）
	b.setoverflow(t, ovf)
	return ovf
}

func (h *hmap) createOverflow() {
	if h.extra == nil {
		h.extra = new(mapextra)
	}
	if h.extra.overflow == nil {
		h.extra.overflow = new([]*bmap)
	}
}

func makemap64(t *maptype, hint int64, h *hmap) *hmap {
	if int64(int(hint)) != hint {
		hint = 0
	}
	return makemap(t, int(hint), h)
}

//对于不指定初始化大小，和初始化值hint<=8（bucketCnt）时，
//go会调用makemap_small函数（源码位置src/runtime/map.go），并直接从堆上进行分配。
func makemap_small() *hmap {
	h := new(hmap)
	h.hash0 = fastrand()
	return h
}

// 如果编译器认为map和第一个bucket可以直接创建在栈上，h和bucket可能都是非空
// 如果h != nil，那么map可以直接在h中创建
// 如果h.buckets != nil，那么h指向的bucket可以作为map的第一个bucket使用
func makemap(t *maptype, hint int, h *hmap) *hmap {
	// math.MulUintptr返回hint与t.bucket.size的乘积，并判断该乘积是否溢出。
	mem, overflow := math.MulUintptr(uintptr(hint), t.bucket.size)
	if overflow || mem > maxAlloc {
		hint = 0
	}

	// initialize Hmap
	if h == nil {
		h = new(hmap)
	}
	// 通过fastrand得到哈希种子
	h.hash0 = fastrand()

	// 根据输入的元素个数hint，找到能装下这些元素的B值
	B := uint8(0)
	for overLoadFactor(hint, B) {
		B++
	}
	h.B = B

	// 分配初始哈希表
	// 如果B为0，那么buckets字段后续会在mapassign方法中lazily分配
	if h.B != 0 {
		var nextOverflow *bmap
		// makeBucketArray创建一个map的底层保存buckets的数组，它最少会分配h.B^2的大小。
		h.buckets, nextOverflow = makeBucketArray(t, h.B, nil)
		if nextOverflow != nil {
			h.extra = new(mapextra)
			h.extra.nextOverflow = nextOverflow
		}
	}

	return h
}

// makeBucket为map创建用于保存buckets的数组。
func makeBucketArray(t *maptype, b uint8, dirtyalloc unsafe.Pointer) (buckets unsafe.Pointer, nextOverflow *bmap) {
	base := bucketShift(b)
	nbuckets := base
	// 对于小的b值（小于4），即桶的数量小于16时，使用溢出桶的可能性很小。对于此情况，就避免计算开销。
	if b >= 4 {
		// 当桶的数量大于等于16个时，正常情况下就会额外创建2^(b-4)个溢出桶
		nbuckets += bucketShift(b - 4)
		sz := t.bucket.size * nbuckets
		up := roundupsize(sz)
		if up != sz {
			nbuckets = up / t.bucket.size
		}
	}
	// 这里，dirtyalloc分两种情况。
	// 1.如果它为nil，则会分配一个新的底层数组。如果它不为nil，则它指向的是曾经分配过的底层数组，该底层数组是由之前同样的t和b参数通过makeBucketArray分配的，
	// 2.如果数组不为空，需要把该数组之前的数据清空并复用。
	if dirtyalloc == nil {
		buckets = newarray(t.bucket, int(nbuckets))
	} else {
		buckets = dirtyalloc
		size := t.bucket.size * nbuckets
		if t.bucket.ptrdata != 0 {
			memclrHasPointers(buckets, size)
		} else {
			memclrNoHeapPointers(buckets, size)
		}
	}

	// 即b大于等于4的情况下，会预分配一些溢出桶。
	// 为了把跟踪这些溢出桶的开销降至最低，使用了以下约定：

	// 如果预分配的溢出桶的overflow指针为nil，那么可以通过指针碰撞（bumping the pointer）获得更多可用桶。
	// （关于指针碰撞：
	// 假设内存是绝对规整的，所有用过的内存都放在一边，空闲的内存放在另一边，中间放着一个指针作为分界点的指示器，
	// 那所分配内存就仅仅是把那个指针向空闲空间那边挪动一段与对象大小相等的距离，
	// 这种分配方式称为“指针碰撞”）

	// 对于最后一个溢出桶，需要一个安全的非nil指针指向它。
	if base != nbuckets {
		nextOverflow = (*bmap)(add(buckets, base*uintptr(t.bucketsize)))
		last := (*bmap)(add(buckets, (nbuckets-1)*uintptr(t.bucketsize)))
		last.setoverflow(t, (*bmap)(buckets)) //最后一个溢出桶又连回第一个桶？
	}

	// 根据上述代码，我们能确定在正常情况下，
	// 正常桶和溢出桶在内存中的存储空间是连续的，只是被 hmap 中的不同字段引用而已。
	return buckets, nextOverflow
}


func mapaccess1(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	// 如果开启了竞态检测 -race
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		pc := funcPC(mapaccess1)
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	// 如果开启了memory sanitizer -msan
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	// 如果map为空或者元素个数为0，返回零值
	if h == nil || h.count == 0 {
		if t.hashMightPanic() {
			t.hasher(key, 0) // see issue 23734
		}
		return unsafe.Pointer(&zeroVal[0])
	}
	// 注意，这里是按位与操作
	// 当h.flags对应的值为hashWriting（代表有其他goroutine正在往map中写key）时，那么位计算的结果不为0，因此抛出以下错误。
	// 这也表明，go的map是非并发安全的
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	// 不同类型的key，会使用不同的hash算法，可详见src/runtime/alg.go中typehash函数中的逻辑
	hash := t.hasher(key, uintptr(h.hash0))
	m := bucketMask(h.B)
	// 按位与操作，找到对应的bucket(低位转换)
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
	// 如果oldbuckets不为空，那么证明map发生了扩容
	// 如果有扩容发生，老的buckets中的数据可能还未搬迁至新的buckets里
	// 所以需要先在老的buckets中找
	if c := h.oldbuckets; c != nil {
		if !h.sameSizeGrow() {
			// There used to be half as many buckets; mask down one more power of two.
			m >>= 1
		}
		oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
		// 如果在oldbuckets中tophash[0]的值，为evacuatedX、evacuatedY，evacuatedEmpty其中之一
		// 则evacuated()返回为true，代表搬迁完成。
		// 因此，只有当搬迁未完成时，才会从此oldbucket中遍历
		if !evacuated(oldb) {
			b = oldb
		}
	}
	// 取出当前key值的tophash值
	top := tophash(hash)
	// 以下是查找的核心逻辑
	// 双重循环遍历：外层循环是从桶到溢出桶遍历；内层是桶中的cell遍历

	// 疑问：这个溢出桶是把整个map作为一块大的内存，并不是作为单个桶的溢出桶？ -作为单个桶bmap

	// 跳出循环的条件有三种：第一种是已经找到key值；第二种是当前桶再无溢出桶；
	// 第三种是当前桶中有cell位的tophash值是emptyRest，
	// 这个值在前面解释过，它代表此时的桶后面的cell还未利用，所以无需再继续遍历。
bucketloop:
	for ; b != nil; b = b.overflow(t) {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			// 因为在bucket中key是用连续的存储空间存储的，因此可以通过bucket地址+数据偏移量（bmap结构体的大小）+ keysize的大小，得到k的地址
			// 同理，value的地址也是相似的计算方法，只是再要加上8个keysize的内存地址
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {
				k = *((*unsafe.Pointer)(k))
			}
			// 判断key是否相等
			if t.key.equal(key, k) {
				e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				if t.indirectelem() {
					e = *((*unsafe.Pointer)(e))
				}
				return e
			}
		}
	}
	// 所有的bucket都未找到，则返回零值
	return unsafe.Pointer(&zeroVal[0])
}

func mapaccess2(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, bool) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		pc := funcPC(mapaccess2)
		racereadpc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	if h == nil || h.count == 0 {
		if t.hashMightPanic() {
			t.hasher(key, 0) // see issue 23734
		}
		return unsafe.Pointer(&zeroVal[0]), false
	}
	if h.flags&hashWriting != 0 {
		throw("concurrent map read and map write")
	}
	hash := t.hasher(key, uintptr(h.hash0))
	m := bucketMask(h.B)
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		if !h.sameSizeGrow() {
			// There used to be half as many buckets; mask down one more power of two.
			m >>= 1
		}
		oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := tophash(hash)
bucketloop:
	for ; b != nil; b = b.overflow(t) {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {
				k = *((*unsafe.Pointer)(k))
			}
			if t.key.equal(key, k) {
				e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				if t.indirectelem() {
					e = *((*unsafe.Pointer)(e))
				}
				return e, true
			}
		}
	}
	return unsafe.Pointer(&zeroVal[0]), false
}

// returns both key and elem. Used by map iterator
func mapaccessK(t *maptype, h *hmap, key unsafe.Pointer) (unsafe.Pointer, unsafe.Pointer) {
	if h == nil || h.count == 0 {
		return nil, nil
	}
	hash := t.hasher(key, uintptr(h.hash0))
	m := bucketMask(h.B)
	b := (*bmap)(add(h.buckets, (hash&m)*uintptr(t.bucketsize)))
	if c := h.oldbuckets; c != nil {
		if !h.sameSizeGrow() {
			// There used to be half as many buckets; mask down one more power of two.
			m >>= 1
		}
		oldb := (*bmap)(add(c, (hash&m)*uintptr(t.bucketsize)))
		// 注：这里是由于迭代时候调用mapaccessK(),所以肯定是发生了扩容
		if !evacuated(oldb) {
			b = oldb
		}
	}
	top := tophash(hash)
bucketloop:
	for ; b != nil; b = b.overflow(t) {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {
				k = *((*unsafe.Pointer)(k))
			}
			if t.key.equal(key, k) {
				e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				if t.indirectelem() {
					e = *((*unsafe.Pointer)(e))
				}
				return k, e
			}
		}
	}
	return nil, nil
}

func mapaccess1_fat(t *maptype, h *hmap, key, zero unsafe.Pointer) unsafe.Pointer {
	e := mapaccess1(t, h, key)
	if e == unsafe.Pointer(&zeroVal[0]) {
		return zero
	}
	return e
}

func mapaccess2_fat(t *maptype, h *hmap, key, zero unsafe.Pointer) (unsafe.Pointer, bool) {
	e := mapaccess1(t, h, key)
	if e == unsafe.Pointer(&zeroVal[0]) {
		return zero, false
	}
	return e, true
}

// Like mapaccess, but allocates a slot for the key if it is not present in the map.
func mapassign(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	// 如果h是空指针，赋值会引起panic
	// 例如以下语句
	// var m map[string]int
	// m["k"] = 1
	if h == nil {
		panic(plainError("assignment to entry in nil map"))
	}
	// 如果开启了竞态检测 -race
	if raceenabled {
		callerpc := getcallerpc()
		pc := funcPC(mapassign)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	// 如果开启了memory sanitizer -msan
	if msanenabled {
		msanread(key, t.key.size)
	}
	// 有其他goroutine正在往map中写key，会抛出以下错误
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}
	// 通过key和哈希种子，算出对应哈希值
	hash := t.hasher(key, uintptr(h.hash0))

	// 将flags的值与hashWriting做按位或运算
	// 因为在当前goroutine可能还未完成key的写入，再次调用t.hasher会发生panic。
	h.flags ^= hashWriting

	if h.buckets == nil {
		h.buckets = newobject(t.bucket) // newarray(t.bucket, 1)
	}

again:
	// bucketMask返回值是2的B次方减1
	// 因此，通过hash值与bucketMask返回值做按位与操作，返回的在buckets数组中的第几号桶
	bucket := hash & bucketMask(h.B)
	// 如果map正在搬迁（即h.oldbuckets != nil）中,则先进行搬迁工作。
	if h.growing() {
		growWork(t, h, bucket)
	}
	// 计算出上面求出的第几号bucket的内存位置
	// post = start + bucketNumber * bucketsize
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.bucketsize)))
	top := tophash(hash)

	var inserti *uint8
	var insertk unsafe.Pointer
	var elem unsafe.Pointer
bucketloop:
	for {
		// 遍历桶中的8个cell
		for i := uintptr(0); i < bucketCnt; i++ {
			// 这里分两种情况，第一种情况是cell位的tophash值和当前tophash值不相等
			// 在 b.tophash[i] != top 的情况下
			// 理论上有可能会是一个空槽位

			// 一般情况下 map 的槽位分布是这样的，e 表示 empty:
			// [h0][h1][h2][h3][h4][e][e][e]
			// 但在执行过 delete 操作时，可能会变成这样:
			// [h0][h1][e][e][h5][e][e][e]

			// 所以如果再插入的话，会尽量往前面的位置插
			// [h0][h1][e][e][h5][e][e][e]
			//          ^
			//          ^
			//       这个位置
			// 所以在循环的时候还要顺便把前面的空位置先记下来
			// 因为有可能在后面会找到相等的key，也可能找不到相等的key
			if b.tophash[i] != top {
				// 记录第一个空的位置
				if isEmpty(b.tophash[i]) && inserti == nil {
					inserti = &b.tophash[i]
					insertk = add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
					elem = add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
				}
				if b.tophash[i] == emptyRest {
					break bucketloop
				}
				continue
			}
			// 第二种情况是cell位的tophash值和当前的tophash值相等
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			if t.indirectkey() {
				k = *((*unsafe.Pointer)(k))
			}
			// 注意，即使当前cell位的tophash值相等，不一定它对应的key也是相等的，所以还要做一个key值判断
			if !t.key.equal(key, k) {
				continue
			}
			// 如果已经有该key了，就更新它
			if t.needkeyupdate() {
				typedmemmove(t.key, k, key)
			}
			// 这里获取到了要插入key对应的value的内存地址
			// pos = start + dataOffset + 8*keysize + i*elemsize
			elem = add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
			// 如果顺利到这，就直接跳到done的结束逻辑中去
			goto done
		}
		// 如果桶中的8个cell遍历完，还未找到对应的空cell或覆盖cell，那么就进入它的溢出桶中去遍历
		ovf := b.overflow(t)
		// 如果连溢出桶中都没有找到合适的cell，跳出循环。
		if ovf == nil {
			break
		}
		b = ovf
	}

	// 在已有的桶和溢出桶中都未找到合适的cell供key写入，那么有可能会触发以下两种情况
	// 情况一：
	// 1.判断当前map的装载因子是否达到设定的6.5阈值，
	// 2.或者当前map的溢出桶数量是否过多。
	// 如果存在这两种情况之一，则进行扩容操作。

	// hashGrow()实际并未完成扩容，对哈希表数据的搬迁（复制）操作是通过growWork()来完成的。
	// 重新跳入again逻辑，在进行完growWork()操作后，再次遍历新的桶。
	if !h.growing() && (overLoadFactor(h.count+1, h.B) || tooManyOverflowBuckets(h.noverflow, h.B)) {
		hashGrow(t, h)
		goto again // Growing the table invalidates everything, so try again
	}

	// 情况二：
	// 在不满足情况一的条件下，会为当前桶再新建溢出桶，并将tophash，key插入到新建溢出桶的对应内存的0号位置
	if inserti == nil {
		// The current bucket and all the overflow buckets connected to it are full, allocate a new one.
		newb := h.newoverflow(t, b)
		inserti = &newb.tophash[0]
		insertk = add(unsafe.Pointer(newb), dataOffset)
		elem = add(insertk, bucketCnt*uintptr(t.keysize))
	}

	// store new key/elem at insert position
	if t.indirectkey() {
		kmem := newobject(t.key)
		*(*unsafe.Pointer)(insertk) = kmem
		insertk = kmem
	}
	if t.indirectelem() {
		vmem := newobject(t.elem)
		*(*unsafe.Pointer)(elem) = vmem
	}
	typedmemmove(t.key, insertk, key)
	// 更新新的key的hash值
	*inserti = top
	// 更新元素数量
	h.count++

done:
	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
	if t.indirectelem() {
		elem = *((*unsafe.Pointer)(elem))
	}
	return elem
}

func mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		pc := funcPC(mapdelete)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
		raceReadObjectPC(t.key, key, callerpc, pc)
	}
	if msanenabled && h != nil {
		msanread(key, t.key.size)
	}
	//如果 hmap 没有初始化，则直接返回
	if h == nil || h.count == 0 {
		if t.hashMightPanic() {
			t.hasher(key, 0) // see issue 23734
		}
		return
	}
	//如果正在进行写入，则 throw
	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}
	//算出当前key 的hash
	hash := t.hasher(key, uintptr(h.hash0))

	// Set hashWriting after calling t.hasher, since t.hasher may panic,
	// in which case we have not actually done a write (delete).
	h.flags ^= hashWriting

	bucket := hash & bucketMask(h.B)
	if h.growing() {
		growWork(t, h, bucket)
	}
	b := (*bmap)(add(h.buckets, bucket*uintptr(t.bucketsize)))
	bOrig := b
	//找到对应的 top
	top := tophash(hash)
search:
	// 遍历当前的 bucket 以及 overflow
	for ; b != nil; b = b.overflow(t) {
		for i := uintptr(0); i < bucketCnt; i++ {
			if b.tophash[i] != top {
				//如果是 emptyReset，说明之后都不存在了，直接break
				if b.tophash[i] == emptyRest {
					break search
				}
				continue
			}
			//找到了对应的位置
			k := add(unsafe.Pointer(b), dataOffset+i*uintptr(t.keysize))
			k2 := k
			if t.indirectkey() {
				k2 = *((*unsafe.Pointer)(k2))
			}
			if !t.key.equal(key, k2) {
				continue
			}
			// Only clear key if there are pointers in it.
			// 这里清理空间 key 的空间
			if t.indirectkey() {
				*(*unsafe.Pointer)(k) = nil
			} else if t.key.ptrdata != 0 {
				memclrHasPointers(k, t.key.size)
			}
			e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+i*uintptr(t.elemsize))
			if t.indirectelem() {
				*(*unsafe.Pointer)(e) = nil
			} else if t.elem.ptrdata != 0 {
				memclrHasPointers(e, t.elem.size)
			} else {
				memclrNoHeapPointers(e, t.elem.size)
			}
			// emptyOne表示曾经有过，然后被清空了
			b.tophash[i] = emptyOne

			// ----这里判断当前位置之后是否还有数据存储过----
			if i == bucketCnt-1 {
				if b.overflow(t) != nil && b.overflow(t).tophash[0] != emptyRest {
					goto notLast
				}
			} else {
				if b.tophash[i+1] != emptyRest {
					goto notLast
				}
			}
			// 到这里就说明当前的 bucket 的 topIndex 以及之后索引包括 overflow 都没有数据过，
			// 准备从后向前进行一波整理
			for {
				b.tophash[i] = emptyRest
				if i == 0 { //则将当前的 top 设置为 emptyRest
					if b == bOrig { // 回到初始桶，结束
						break
					}
					//找到当前 bucket 的上一个 bucket
					c := b
					for b = bOrig; b.overflow(t) != c; b = b.overflow(t) {
					}
					i = bucketCnt - 1
				} else {	//寻找上一个top
					i--
				}
				if b.tophash[i] != emptyOne {
					break
				}
			}
		notLast:
			h.count--
			// 重置哈希种子，使攻击者更难攻击
			// repeatedly trigger hash collisions. See issue 25237.
			if h.count == 0 {
				h.hash0 = fastrand()
			}
			break search
		}
	}

	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}

// mapiterinit initializes the hiter struct used for ranging over maps.
// The hiter struct pointed to by 'it' is allocated on the stack
// by the compilers order pass or on the heap by reflect_mapiterinit.
// Both need to have zeroed hiter since the struct contains pointers.
func mapiterinit(t *maptype, h *hmap, it *hiter) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapiterinit))
	}

	if h == nil || h.count == 0 { //如果 hmap 不存在或者没有数量，则直接返回
		return
	}

	if unsafe.Sizeof(hiter{})/sys.PtrSize != 12 {
		throw("hash_iter size incorrect") // see cmd/compile/internal/reflectdata/reflect.go
	}
	it.t = t
	it.h = h

	// grab snapshot of bucket state
	it.B = h.B
	it.buckets = h.buckets
	if t.bucket.ptrdata == 0 {
		// Allocate the current slice and remember pointers to both current and old.
		// This preserves all relevant overflow buckets alive even if
		// the table grows and/or overflow buckets are added to the table
		// while we are iterating.
		// 初始化数据
		h.createOverflow()
		it.overflow = h.extra.overflow
		it.oldoverflow = h.extra.oldoverflow
	}

	// decide where to start
	r := uintptr(fastrand())	// 从这里可以看出，起始位置上是一个随机数
	if h.B > 31-bucketCntBits {
		r += uintptr(fastrand()) << 31
	}
	it.startBucket = r & bucketMask(h.B) 	//找到随机开始的 bucketIndex
	it.offset = uint8(r >> h.B & (bucketCnt - 1))	// 找到随机开始的 bucket 的 topHash 的索引

	// iterator state
	it.bucket = it.startBucket

	// Remember we have an iterator.
	// Can run concurrently with another mapiterinit().
	if old := h.flags; old&(iterator|oldIterator) != iterator|oldIterator {
		atomic.Or8(&h.flags, iterator|oldIterator)	//将 flags 变成迭代的情况
	}

	mapiternext(it)
}

func mapiternext(it *hiter) {
	h := it.h
	if raceenabled {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(mapiternext))
	}
	if h.flags&hashWriting != 0 {	//如果正在并发写，则 throw
		throw("concurrent map iteration and map write")
	}
	t := it.t
	bucket := it.bucket	//这里的 bucket 是 bucketIndex
	b := it.bptr
	i := it.i
	checkBucket := it.checkBucket

next:
	if b == nil {
		// wrappd 为 true 表示已经到过最后
		// bucket == it.startBucket 表示已经一个循环了
		if bucket == it.startBucket && it.wrapped {
			// end of iteration
			it.key = nil
			it.elem = nil
			return
		}
		if h.growing() && it.B == h.B { //如果表示正在迁移的途中,因此要也要遍历老的部分
			oldbucket := bucket & it.h.oldbucketmask()
			b = (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.bucketsize)))
			if !evacuated(b) {	//如果老的部分未迁移，还要遍历现在的 bucketIndex
				checkBucket = bucket
			} else {
				b = (*bmap)(add(it.buckets, bucket*uintptr(t.bucketsize)))
				checkBucket = noCheck
			}
		} else {
			b = (*bmap)(add(it.buckets, bucket*uintptr(t.bucketsize)))
			checkBucket = noCheck
		}
		bucket++
		if bucket == bucketShift(it.B) {
			bucket = 0
			it.wrapped = true
		}
		i = 0
	}
	for ; i < bucketCnt; i++ {	//遍历当前 bucket 的 8 个 key
		offi := (i + it.offset) & (bucketCnt - 1)
		if isEmpty(b.tophash[offi]) || b.tophash[offi] == evacuatedEmpty {	//如果为空或者是已经迁移了，则跳过
			// TODO: emptyRest is hard to use here, as we start iterating
			// in the middle of a bucket. It's feasible, just tricky.
			continue
		}
		k := add(unsafe.Pointer(b), dataOffset+uintptr(offi)*uintptr(t.keysize))
		if t.indirectkey() {
			k = *((*unsafe.Pointer)(k))
		}
		e := add(unsafe.Pointer(b), dataOffset+bucketCnt*uintptr(t.keysize)+uintptr(offi)*uintptr(t.elemsize))
		//如果 oldBucket 还存在，且非等量迁移
		if checkBucket != noCheck && !h.sameSizeGrow() {
			if t.reflexivekey() || t.key.equal(k, k) {
				hash := t.hasher(k, uintptr(h.hash0))
				// 注：这里只有hash种子不一样才有可能！=
				if hash&bucketMask(it.B) != checkBucket {	//如果不是相同的 key ，则跳过
					continue
				}
			} else {	// 这里处理的是 NaN 的情况
				if checkBucket>>(it.B-1) != uintptr(b.tophash[offi]&1) {
					continue
				}
			}
		}
		// 这里表示没有进行迁移（不论是在 oldbuckets 还是 buckets）以及 NaN 的情况，都能遍历出来

		// 注：这里b.tophash[offi] != evacuatedX如果按照正常扩容逻辑if !evacuated(b)是不会==的，
		// 扩容是按照桶为基本单位的，
		// 可能在遍历桶第一元素时，是没有发生扩容的，
		// 但是在遍历桶第二个元素时，发生了扩容。
		// 所以，要按照b.tophash[offi] != evacuatedX和b.tophash[offi] == evacuatedX情况分开讨论
		if (b.tophash[offi] != evacuatedX && b.tophash[offi] != evacuatedY) ||
			!(t.reflexivekey() || t.key.equal(k, k)) {
			it.key = k
			if t.indirectelem() {
				e = *((*unsafe.Pointer)(e))
			}
			it.elem = e
		} else {	// 这里表示非 NaN 且正在迁移的部分
			rk, re := mapaccessK(t, h, k)	//从当前的key 对应的 oldBucket 或 bucket 寻找数据
			if rk == nil {
				continue // key has been deleted
			}
			it.key = rk
			it.elem = re
		}
		// 指向下一个桶
		it.bucket = bucket
		if it.bptr != b { // avoid unnecessary write barrier; see issue 14921
			it.bptr = b
		}
		it.i = i + 1
		it.checkBucket = checkBucket
		return
	}
	b = b.overflow(t)	//获取当前bucket 的overflow
	i = 0
	goto next
}

// mapclear deletes all keys from a map.
func mapclear(t *maptype, h *hmap) {
	if raceenabled && h != nil {
		callerpc := getcallerpc()
		pc := funcPC(mapclear)
		racewritepc(unsafe.Pointer(h), callerpc, pc)
	}

	if h == nil || h.count == 0 {
		return
	}

	if h.flags&hashWriting != 0 {
		throw("concurrent map writes")
	}

	h.flags ^= hashWriting	// "写入位"置1

	h.flags &^= sameSizeGrow //将 sameSizeGrow 清0
	h.oldbuckets = nil
	h.nevacuate = 0
	h.noverflow = 0
	h.count = 0

	// Reset the hash seed to make it more difficult for attackers to
	// repeatedly trigger hash collisions. See issue 25237.
	h.hash0 = fastrand()

	// 这里直接数据清空
	if h.extra != nil {
		*h.extra = mapextra{}
	}

	//将其中buckets数据清空，并拿到nextOverFlow
	_, nextOverflow := makeBucketArray(t, h.B, h.buckets)
	if nextOverflow != nil {
		// If overflow buckets are created then h.extra
		// will have been allocated during initial bucket creation.
		h.extra.nextOverflow = nextOverflow
	}

	if h.flags&hashWriting == 0 {
		throw("concurrent map writes")
	}
	h.flags &^= hashWriting
}

func hashGrow(t *maptype, h *hmap) {

	bigger := uint8(1)
	if !overLoadFactor(h.count+1, h.B) {
		bigger = 0
		h.flags |= sameSizeGrow
	}
	// 记录老的buckets
	oldbuckets := h.buckets
	// 申请新的buckets空间
	newbuckets, nextOverflow := makeBucketArray(t, h.B+bigger, nil)
	// 注意&^ 运算符，这块代码的逻辑是转移标志位
	flags := h.flags &^ (iterator | oldIterator)
	if h.flags&iterator != 0 {
		flags |= oldIterator
	}
	// 提交grow (atomic wrt gc)
	h.B += bigger	// 注：如果是增量扩容，这里就已经预先+1
	h.flags = flags
	h.oldbuckets = oldbuckets
	h.buckets = newbuckets
	// 搬迁进度为0
	h.nevacuate = 0
	// overflow buckets 数为0
	h.noverflow = 0

	// 如果发现hmap是通过extra字段 来存储 overflow buckets时
	if h.extra != nil && h.extra.overflow != nil {
		// Promote current overflow buckets to the old generation.
		if h.extra.oldoverflow != nil {
			throw("oldoverflow is not nil")
		}
		h.extra.oldoverflow = h.extra.overflow
		h.extra.overflow = nil
	}
	if nextOverflow != nil {
		if h.extra == nil {
			h.extra = new(mapextra)
		}
		h.extra.nextOverflow = nextOverflow
	}

	// the actual copying of the hash table data is done incrementally
	// by growWork() and evacuate().
}

// overLoadFactor reports whether count items placed in 1<<B buckets is over loadFactor.
func overLoadFactor(count int, B uint8) bool {
	return count > bucketCnt && uintptr(count) > loadFactorNum*(bucketShift(B)/loadFactorDen)
}

// tooManyOverflowBuckets reports whether noverflow buckets is too many for a map with 1<<B buckets.
// Note that most of these overflow buckets must be in sparse use;
// if use was dense, then we'd have already triggered regular map growth.
func tooManyOverflowBuckets(noverflow uint16, B uint8) bool {
	// If the threshold is too low, we do extraneous work.
	// If the threshold is too high, maps that grow and shrink can hold on to lots of unused memory.
	// "too many" means (approximately) as many overflow buckets as regular buckets.
	// See incrnoverflow for more details.
	if B > 15 {
		B = 15
	}
	// The compiler doesn't see here that B < 16; mask B to generate shorter shift code.
	return noverflow >= uint16(1)<<(B&15)
}

// growing reports whether h is growing. The growth may be to the same size or bigger.
func (h *hmap) growing() bool {
	return h.oldbuckets != nil
}

// sameSizeGrow reports whether the current growth is to a map of the same size.
func (h *hmap) sameSizeGrow() bool {
	return h.flags&sameSizeGrow != 0
}

// noldbuckets calculates the number of buckets prior to the current map growth.
func (h *hmap) noldbuckets() uintptr {
	oldB := h.B
	if !h.sameSizeGrow() {
		oldB--	//注：在hasGrow()中，对增量扩容B=B+1，所以这里要-1，得到原始容量
	}
	return bucketShift(oldB)
}

// oldbucketmask provides a mask that can be applied to calculate n % noldbuckets().
func (h *hmap) oldbucketmask() uintptr {
	return h.noldbuckets() - 1
}

func growWork(t *maptype, h *hmap, bucket uintptr) {
	// 为了确认搬迁的 bucket 是我们正在使用的 bucket
	// 即如果当前key映射到老的bucket1，那么就搬迁该bucket1。
	evacuate(t, h, bucket&h.oldbucketmask())

	// 如果还未完成扩容工作，则再搬迁一个bucket。
	if h.growing() {
		evacuate(t, h, h.nevacuate)
	}
}

func bucketEvacuated(t *maptype, h *hmap, bucket uintptr) bool {
	b := (*bmap)(add(h.oldbuckets, bucket*uintptr(t.bucketsize)))
	return evacuated(b)
}

// evacDst is an evacuation destination.
type evacDst struct {
	b *bmap          // 指向的是当前桶，如果超过8个键值对则指向的是溢出桶
	i int            // 当前桶存放的键值对数量
	k unsafe.Pointer // 指向下一个空的key数组
	e unsafe.Pointer // 指向下一个空的value数组
}

func evacuate(t *maptype, h *hmap, oldbucket uintptr) {
	// 首先定位老的bucket的地址
	b := (*bmap)(add(h.oldbuckets, oldbucket*uintptr(t.bucketsize)))
	// newbit代表扩容之前老的bucket个数
	newbit := h.noldbuckets()
	// 判断该bucket是否已经被搬迁
	// 注：因为扩容是在对数据修改的操作基础之上的，并且每次尝试两次扩容（1.当前key的bucket 2.扩容进度的bucket）
	// 那么就肯定有已经扩容的bucket，就跳过
	if !evacuated(b) {
		// TODO: reuse overflow buckets instead of using new ones, if there
		// xy 包含了高低区间的搬迁目的地内存信息
		// x.b 是对应的搬迁目的桶
		// x.k 是指向对应目的桶中存储当前key的内存地址
		// x.e 是指向对应目的桶中存储当前value的内存地址
		var xy [2]evacDst
		x := &xy[0]
		x.b = (*bmap)(add(h.buckets, oldbucket*uintptr(t.bucketsize)))
		x.k = add(unsafe.Pointer(x.b), dataOffset)
		x.e = add(x.k, bucketCnt*uintptr(t.keysize))
		// 只有当增量扩容时才计算bucket y的相关信息（和后续计算useY相呼应）
		if !h.sameSizeGrow() {
			// Only calculate y pointers if we're growing bigger.
			// Otherwise GC can see bad pointers.
			y := &xy[1]
			y.b = (*bmap)(add(h.buckets, (oldbucket+newbit)*uintptr(t.bucketsize)))
			y.k = add(unsafe.Pointer(y.b), dataOffset)
			y.e = add(y.k, bucketCnt*uintptr(t.keysize))
		}

		// evacuate 函数每次只完成一个 bucket 的搬迁工作，因此要遍历完此 bucket 的所有的 cell，将有值的 cell copy 到新的地方。
		// bucket 还会链接 overflow bucket，它们同样需要搬迁。
		// 因此同样会有 2 层循环，外层遍历 bucket 和 overflow bucket，内层遍历 bucket 的所有 cell。

		// 遍历当前桶bucket和其之后的溢出桶overflow bucket
		// 注意：初始的b是待搬迁的老bucket
		for ; b != nil; b = b.overflow(t) {
			k := add(unsafe.Pointer(b), dataOffset)
			e := add(k, bucketCnt*uintptr(t.keysize))
			// 遍历桶中的cell，i，k，e分别用于对应tophash，key和value
			for i := 0; i < bucketCnt; i, k, e = i+1, add(k, uintptr(t.keysize)), add(e, uintptr(t.elemsize)) {
				top := b.tophash[i]
				// 如果当前cell的tophash值是emptyOne或者emptyRest，则代表此cell没有key。并将其标记为evacuatedEmpty，表示它“已经被搬迁”。
				if isEmpty(top) {
					b.tophash[i] = evacuatedEmpty
					continue
				}
				// 正常不会出现这种情况
				// 未被搬迁的 cell 只可能是emptyOne、emptyRest或是正常的 top hash（大于等于 minTopHash）
				if top < minTopHash {
					throw("bad map state")
				}
				k2 := k
				// 如果 key 是指针，则解引用
				if t.indirectkey() {
					k2 = *((*unsafe.Pointer)(k2))
				}
				var useY uint8
				// 如果是增量扩容
				if !h.sameSizeGrow() {
					// 计算哈希值，判断当前key和vale是要被搬迁到bucket x还是bucket y
					hash := t.hasher(k2, uintptr(h.hash0))
					if h.flags&iterator != 0 && !t.reflexivekey() && !t.key.equal(k2, k2) {
						// 有一个特殊情况：有一种 key，每次对它计算 hash，得到的结果都不一样。
						// 这个 key 就是 math.NaN() 的结果，它的含义是 not a number，类型是 float64。
						// 当它作为 map 的 key时，会遇到一个问题：再次计算它的哈希值和它当初插入 map 时的计算出来的哈希值不一样！
						// 这个 key 是永远不会被 Get 操作获取的！当使用 m[math.NaN()] 语句的时候，是查不出来结果的。
						// 这个 key 只有在遍历整个 map 的时候，才能被找到。
						// 并且，可以向一个 map 插入多个数量的 math.NaN() 作为 key，它们并不会被互相覆盖。
						// 1.当搬迁碰到 math.NaN() 的 key 时，只通过 tophash 的最低位决定分配到 X part 还是 Y part（如果扩容后是原来 buckets 数量的 2 倍）。
						// 2.如果 tophash 的最低位是 0 ，分配到 X part；如果是 1 ，则分配到 Y part。
						useY = top & 1
						top = tophash(hash)
						// 对于正常key，进入以下else逻辑
					} else {
						// newbit正好是最高位的位数
						if hash&newbit != 0 {
							useY = 1
						}
					}
				}

				if evacuatedX+1 != evacuatedY || evacuatedX^1 != evacuatedY {
					throw("bad evacuatedN")
				}

				b.tophash[i] = evacuatedX + useY // evacuatedX + 1 == evacuatedY
				// useY要么为0，要么为1。
				// 这里就是选取在bucket x的起始内存位置，或者选择在bucket y的起始内存位置（只有增量同步才会有这个选择可能）。
				dst := &xy[useY]                 // evacuation destination

				// 如果目的地的桶已经装满了（8个cell），那么需要新建一个溢出桶，继续搬迁到溢出桶上去。
				// 注：这里有溢出桶是因为原位置就有溢出桶
				if dst.i == bucketCnt {
					dst.b = h.newoverflow(t, dst.b)
					dst.i = 0
					dst.k = add(unsafe.Pointer(dst.b), dataOffset)
					dst.e = add(dst.k, bucketCnt*uintptr(t.keysize))
				}
				dst.b.tophash[dst.i&(bucketCnt-1)] = top // mask dst.i as an optimization, to avoid a bounds check
				// 注：dst.i&(bucketCnt-1)这种写法好处是不会数组越界

				// 如果待搬迁的key是指针，则复制指针过去
				if t.indirectkey() {
					*(*unsafe.Pointer)(dst.k) = k2 // copy pointer
				// 如果待搬迁的key是值，则复制值过去
				} else {
					typedmemmove(t.key, dst.k, k) // copy elem
				}
				// value和key同理
				if t.indirectelem() {
					*(*unsafe.Pointer)(dst.e) = *(*unsafe.Pointer)(e)
				} else {
					typedmemmove(t.elem, dst.e, e)
				}
				// 将当前搬迁目的桶的记录key/value的索引值（也可以理解为cell的索引值）加一
				dst.i++

				// 由于桶的内存布局中在最后还有overflow的指针，多以这里不用担心更新有可能会超出key和value数组的指针地址。
				dst.k = add(dst.k, uintptr(t.keysize))
				dst.e = add(dst.e, uintptr(t.elemsize))
			}
		}
		// 如果没有协程在使用老的桶，就对老的桶进行清理，用于帮助gc
		// 注：一个goroutinue在修改，一个goroutinue在遍历
		if h.flags&oldIterator == 0 && t.bucket.ptrdata != 0 {
			b := add(h.oldbuckets, oldbucket*uintptr(t.bucketsize))
			// 只清除bucket 的 key,value 部分，保留 top hash 部分，指示搬迁状态
			ptr := add(b, dataOffset)
			n := uintptr(t.bucketsize) - dataOffset
			memclrHasPointers(ptr, n)
		}
	}
	// 用于更新搬迁进度
	if oldbucket == h.nevacuate {
		advanceEvacuationMark(h, t, newbit)
	}
}

func advanceEvacuationMark(h *hmap, t *maptype, newbit uintptr) {
	// 搬迁桶的进度加一
	h.nevacuate++
	// 实验表明，1024至少会比newbit高出一个数量级（newbit代表扩容之前老的bucket个数）。
	// 所以，用当前进度加上1024用于确保O(1)行为。
	stop := h.nevacuate + 1024
	if stop > newbit {
		stop = newbit
	}
	// 计算已经搬迁完的桶数
	for h.nevacuate != stop && bucketEvacuated(t, h, h.nevacuate) {
		h.nevacuate++
	}
	// 如果h.nevacuate == newbit，则代表所有的桶都已经搬迁完毕
	if h.nevacuate == newbit {
		// 搬迁完毕，所以指向老的buckets的指针置为nil
		h.oldbuckets = nil

		// 在讲解hmap的结构中，有过说明。如果key和value均不包含指针，则都可以inline。
		// 那么保存它们的buckets数组其实是挂在hmap.extra中的。所以，这种情况下，其实我们是搬迁的extra的buckets数组。
		// 因此，在这种情况下，需要在搬迁完毕后，将hmap.extra.oldoverflow指针置为nil。
		if h.extra != nil {
			h.extra.oldoverflow = nil
		}
		// 最后，清除正在扩容的标志位，扩容完毕。
		h.flags &^= sameSizeGrow
	}
}

// Reflect stubs. Called from ../reflect/asm_*.s

//go:linkname reflect_makemap reflect.makemap
func reflect_makemap(t *maptype, cap int) *hmap {
	// Check invariants and reflects math.
	if t.key.equal == nil {
		throw("runtime.reflect_makemap: unsupported map key type")
	}
	if t.key.size > maxKeySize && (!t.indirectkey() || t.keysize != uint8(sys.PtrSize)) ||
		t.key.size <= maxKeySize && (t.indirectkey() || t.keysize != uint8(t.key.size)) {
		throw("key size wrong")
	}
	if t.elem.size > maxElemSize && (!t.indirectelem() || t.elemsize != uint8(sys.PtrSize)) ||
		t.elem.size <= maxElemSize && (t.indirectelem() || t.elemsize != uint8(t.elem.size)) {
		throw("elem size wrong")
	}
	if t.key.align > bucketCnt {
		throw("key align too big")
	}
	if t.elem.align > bucketCnt {
		throw("elem align too big")
	}
	if t.key.size%uintptr(t.key.align) != 0 {
		throw("key size not a multiple of key align")
	}
	if t.elem.size%uintptr(t.elem.align) != 0 {
		throw("elem size not a multiple of elem align")
	}
	if bucketCnt < 8 {
		throw("bucketsize too small for proper alignment")
	}
	if dataOffset%uintptr(t.key.align) != 0 {
		throw("need padding in bucket (key)")
	}
	if dataOffset%uintptr(t.elem.align) != 0 {
		throw("need padding in bucket (elem)")
	}

	return makemap(t, cap, nil)
}

//go:linkname reflect_mapaccess reflect.mapaccess
func reflect_mapaccess(t *maptype, h *hmap, key unsafe.Pointer) unsafe.Pointer {
	elem, ok := mapaccess2(t, h, key)
	if !ok {
		// reflect wants nil for a missing element
		elem = nil
	}
	return elem
}

//go:linkname reflect_mapassign reflect.mapassign
func reflect_mapassign(t *maptype, h *hmap, key unsafe.Pointer, elem unsafe.Pointer) {
	p := mapassign(t, h, key)
	typedmemmove(t.elem, p, elem)
}

//go:linkname reflect_mapdelete reflect.mapdelete
func reflect_mapdelete(t *maptype, h *hmap, key unsafe.Pointer) {
	mapdelete(t, h, key)
}

//go:linkname reflect_mapiterinit reflect.mapiterinit
func reflect_mapiterinit(t *maptype, h *hmap) *hiter {
	it := new(hiter)
	mapiterinit(t, h, it)
	return it
}

//go:linkname reflect_mapiternext reflect.mapiternext
func reflect_mapiternext(it *hiter) {
	mapiternext(it)
}

//go:linkname reflect_mapiterkey reflect.mapiterkey
func reflect_mapiterkey(it *hiter) unsafe.Pointer {
	return it.key
}

//go:linkname reflect_mapiterelem reflect.mapiterelem
func reflect_mapiterelem(it *hiter) unsafe.Pointer {
	return it.elem
}

//go:linkname reflect_maplen reflect.maplen
func reflect_maplen(h *hmap) int {
	if h == nil {
		return 0
	}
	if raceenabled {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(reflect_maplen))
	}
	return h.count
}

//go:linkname reflectlite_maplen internal/reflectlite.maplen
func reflectlite_maplen(h *hmap) int {
	if h == nil {
		return 0
	}
	if raceenabled {
		callerpc := getcallerpc()
		racereadpc(unsafe.Pointer(h), callerpc, funcPC(reflect_maplen))
	}
	return h.count
}

const maxZero = 1024 // must match value in reflect/value.go:maxZero cmd/compile/internal/gc/walk.go:zeroValSize
var zeroVal [maxZero]byte
