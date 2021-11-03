// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync

import (
	"internal/race"
	"sync/atomic"
	"unsafe"
)

// WaitGroup等待一组goroutine完成。
// goroutine通过调用 Add(),增加等待的goroutines的数量。
// goroutines组中的goroutine等执行完成后，会调用Done()。
// 同时，WaitGroup 不要被复制使用。

type WaitGroup struct {
	// noCopy可以嵌入到不能复制的结构中
	// 首次使用后
	noCopy noCopy

	// 64位：高32位是counter, 低32位是等待者的数量。
	// 64位院子操作需要64位对齐，但是32位系统编译器无法保证它，
	// 所以我们使用12字节，然后使用8字节作为状态，其余四个字节用在放信号。
	state1 [3]uint32  // 12字节
}

// state()用来返回，state,sema(信号)。
// 区分32位系统和64为系统
func (wg *WaitGroup) state() (statep *uint64, semap *uint32) {
	// unsafe.Pointer 可以包含任何变量的地址。
	// 64位编译器地址能被8整除，由此可判断是否为64位对齐
	if uintptr(unsafe.Pointer(&wg.state1))%8 == 0 {
		//这里涉及到了对齐。
		return (*uint64)(unsafe.Pointer(&wg.state1)), &wg.state1[2]
	} else {
		return (*uint64)(unsafe.Pointer(&wg.state1[1])), &wg.state1[0]
	}
}

// 增加 delata
// 参考：src/sync/waitgroup_test.go（error示例）
// int(-1)的补码和uint64(1)是一样的，所以可以直接强制转换
func (wg *WaitGroup) Add(delta int) {
	// 获取包含counter与waiter的复合状态statep，表示信号量值的semap
	statep, semap := wg.state()
	if race.Enabled {
		_ = *statep // trigger nil deref early
		if delta < 0 {
			// Synchronize decrements with Wait.
			race.ReleaseMerge(unsafe.Pointer(wg))
		}
		race.Disable()
		defer race.Enable()
	}
	// delta左移动32(低32位是sema)
	state := atomic.AddUint64(statep, uint64(delta)<<32)
	v := int32(state >> 32)	//去掉低位就是值
	w := uint32(state)	// 去掉高位就是，就是等待的数量
	if race.Enabled && delta > 0 && v == int32(delta) {
		// The first increment must be synchronized with Wait.
		// Need to model this as a read, because there can be
		// several concurrent wg.counter transitions from 0.
		race.Read(unsafe.Pointer(semap))
	}
	// 情况1：这是很低级的错误，counter值不能为负

	// 数据溢出了，delta为负数时，减到负数再强转整数会导致数据溢出
	if v < 0 {
		panic("sync: negative WaitGroup counter")
	}
	// 情况2：misuse引起panic
	// 因为wg其实是可以用复用的，但是下一次复用的基础是需要将所有的状态重置为0才可以

	// 添加与等待同时调用，不能再已经等待的时候，才开始添加
	if w != 0 && delta > 0 && v == int32(delta) {
		panic("sync: WaitGroup misuse: Add called concurrently with Wait")
	}

	// 情况3：本次Add操作只负责增加counter值，直接返回即可。
	// 如果此时counter值大于0，唤醒的操作留给之后的Add调用者（执行Add(negative int)）
	// 如果waiter值为0，代表此时还没有阻塞的waiter

	// 正常的运行到这里就结束了
	if v > 0 || w == 0 {
		return
	}

	// 情况4: misuse引起的panic

	// 添加与等待同时调用
	if *statep != state {
		panic("sync: WaitGroup misuse: Add called concurrently with Wait")
	}

	// 如果执行到这，一定是 counter=0，waiter>0
	// 能执行到这，一定是执行了Add(-x)的goroutine
	// 它的执行，代表所有子goroutine已经完成了任务
	// 因此，我们需要将复合状态全部归0，并释放掉waiter个数的信号量
	*statep = 0
	for ; w != 0; w-- {
		// 释放信号量，执行一次就将唤醒一个阻塞的waiter
		runtime_Semrelease(semap, false, 0)
	}
}

// Done decrements the WaitGroup counter by one.
func (wg *WaitGroup) Done() {
	wg.Add(-1)
}

// 注：wait是通过循环CAS来检测数据的变化，这通常意味着数据的变化时间很短
func (wg *WaitGroup) Wait() {
	statep, semap := wg.state()
	if race.Enabled {
		_ = *statep // trigger nil deref early
		race.Disable()
	}
	for {
		state := atomic.LoadUint64(statep)	// 原子读取复合状态statep
		v := int32(state >> 32)				// 获取counter值
		w := uint32(state)					// 获取waiter值
		// 如果此时v==0,证明已经没有待执行任务的子goroutine，直接退出即可。
		if v == 0 {
			// Counter is 0, no need to wait.
			if race.Enabled {
				race.Enable()
				race.Acquire(unsafe.Pointer(wg))
			}
			return
		}
		// 如果在执行CAS原子操作和读取复合状态之间，没有其他goroutine更改了复合状态
		// 那么就将waiter值+1，否则：进入下一轮循环，重新读取复合状态
		if atomic.CompareAndSwapUint64(statep, state, state+1) {
			if race.Enabled && w == 0 {
				race.Write(unsafe.Pointer(semap))
			}
			// 对waiter值累加成功后
			// 等待Add函数中调用 runtime_Semrelease 唤醒自己
			runtime_Semacquire(semap)

			// reused 引发panic
			// 在当前goroutine被唤醒时，由于唤醒自己的goroutine通过调用Add方法时
			// 已经通过 *statep = 0 语句做了重置操作
			// 此时的复合状态位不为0，就是因为还未等Waiter执行完Wait，WaitGroup就已经发生了复用
			if *statep != 0 {
				panic("sync: WaitGroup is reused before previous Wait has returned")
			}
			if race.Enabled {
				race.Enable()
				race.Acquire(unsafe.Pointer(wg))
			}
			return
		}
	}
}
