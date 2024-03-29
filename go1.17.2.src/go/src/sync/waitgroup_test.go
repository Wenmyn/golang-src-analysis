// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"internal/race"
	"runtime"
	. "sync"
	"sync/atomic"
	"testing"
)

func testWaitGroup(t *testing.T, wg1 *WaitGroup, wg2 *WaitGroup) {
	n := 16
	wg1.Add(n)
	wg2.Add(n)
	exited := make(chan bool, n)
	for i := 0; i != n; i++ {
		go func() {
			wg1.Done()
			wg2.Wait()
			exited <- true
		}()
	}
	wg1.Wait()
	for i := 0; i != n; i++ {
		select {
		case <-exited:
			t.Fatal("WaitGroup released group too soon")
		default:
		}
		wg2.Done()
	}
	for i := 0; i != n; i++ {
		<-exited // Will block if barrier fails to unlock someone.
	}
}

func TestWaitGroup(t *testing.T) {
	wg1 := &WaitGroup{}
	wg2 := &WaitGroup{}

	// Run the same test a few times to ensure barrier is in a proper state.
	for i := 0; i != 8; i++ {
		testWaitGroup(t, wg1, wg2)
	}
}

func knownRacy(t *testing.T) {
	if race.Enabled {
		t.Skip("skipping known-racy test under the race detector")
	}
}

func TestWaitGroupMisuse(t *testing.T) {
	defer func() {
		err := recover()
		if err != "sync: negative WaitGroup counter" {
			t.Fatalf("Unexpected panic: %#v", err)
		}
	}()
	wg := &WaitGroup{}
	wg.Add(1)
	wg.Done()
	wg.Done()
	t.Fatal("Should panic")
}

// pollUntilEqual blocks until v, loaded atomically, is
// equal to the target.
func pollUntilEqual(v *uint32, target uint32) {
	for {
		for i := 0; i < 1e3; i++ {
			if atomic.LoadUint32(v) == target {
				return
			}
		}
		// yield to avoid deadlock with the garbage collector
		// see issue #20072
		runtime.Gosched()
	}
}

func TestWaitGroupMisuse2(t *testing.T) {
	knownRacy(t)
	if runtime.NumCPU() <= 4 {
		t.Skip("NumCPU<=4, skipping: this test requires parallelism")
	}
	defer func() {
		err := recover()
		if err != "sync: negative WaitGroup counter" &&
			err != "sync: WaitGroup misuse: Add called concurrently with Wait" &&
			err != "sync: WaitGroup is reused before previous Wait has returned" {
			t.Fatalf("Unexpected panic: %#v", err)
		}
	}()
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	done := make(chan interface{}, 2)
	// The detection is opportunistic, so we want it to panic
	// at least in one run out of a million.
	for i := 0; i < 1e6; i++ {
		var wg WaitGroup
		var here uint32
		wg.Add(1)
		go func() {
			defer func() {
				done <- recover()
			}()
			atomic.AddUint32(&here, 1)
			pollUntilEqual(&here, 3)
			wg.Wait()
		}()
		go func() {
			defer func() {
				done <- recover()
			}()
			atomic.AddUint32(&here, 1)
			pollUntilEqual(&here, 3)
			wg.Add(1) // This is the bad guy. //对应情况3
			wg.Done()
		}()
		atomic.AddUint32(&here, 1)
		pollUntilEqual(&here, 3)
		wg.Done()
		for j := 0; j < 2; j++ {
			if err := <-done; err != nil {
				panic(err)
			}
		}
	}
	t.Fatal("Should panic")
}

func TestWaitGroupMisuse3(t *testing.T) {
	knownRacy(t)
	if runtime.NumCPU() <= 1 {
		t.Skip("NumCPU==1, skipping: this test requires parallelism")
	}
	defer func() {
		err := recover()
		if err != "sync: negative WaitGroup counter" &&
			err != "sync: WaitGroup misuse: Add called concurrently with Wait" &&
			err != "sync: WaitGroup is reused before previous Wait has returned" {
			t.Fatalf("Unexpected panic: %#v", err)
		}
	}()
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	done := make(chan interface{}, 3)
	// The detection is opportunistically, so we want it to panic
	// at least in one run out of a million.
	for i := 0; i < 1e6; i++ {
		var wg WaitGroup
		wg.Add(1)
		go func() {
			defer func() {
				done <- recover()
			}()
			wg.Done()
		}()
		go func() {
			defer func() {
				done <- recover()
			}()
			wg.Wait()
			// Start reusing the wg before waiting for the Wait below to return.
			// 在wait之前add，意味着done永远不会为0，wait会一直阻塞.
			wg.Add(1)
			go func() {
				wg.Done()
			}()
			wg.Wait()
		}()
		go func() {
			defer func() {
				done <- recover()
			}()
			wg.Wait()
		}()
		for j := 0; j < 3; j++ {
			if err := <-done; err != nil {
				panic(err)
			}
		}
	}
	t.Fatal("Should panic")
}

func TestWaitGroupRace(t *testing.T) {
	// Run this test for about 1ms.
	for i := 0; i < 1000; i++ {
		wg := &WaitGroup{}
		n := new(int32)
		// spawn goroutine 1
		wg.Add(1)
		go func() {
			atomic.AddInt32(n, 1)
			wg.Done()
		}()
		// spawn goroutine 2
		wg.Add(1)
		go func() {
			atomic.AddInt32(n, 1)
			wg.Done()
		}()
		// Wait for goroutine 1 and 2
		wg.Wait()
		if atomic.LoadInt32(n) != 2 {
			t.Fatal("Spurious wakeup from Wait")
		}
	}
}

func TestWaitGroupAlign(t *testing.T) {
	type X struct {
		x  byte
		wg WaitGroup
	}
	var x X
	x.wg.Add(1)
	go func(x *X) {
		x.wg.Done()
	}(&x)
	x.wg.Wait()
}

func BenchmarkWaitGroupUncontended(b *testing.B) {
	type PaddedWaitGroup struct {
		WaitGroup
		pad [128]uint8
	}
	b.RunParallel(func(pb *testing.PB) {
		var wg PaddedWaitGroup
		for pb.Next() {
			wg.Add(1)
			wg.Done()
			wg.Wait()
		}
	})
}

func benchmarkWaitGroupAddDone(b *testing.B, localWork int) {
	var wg WaitGroup
	b.RunParallel(func(pb *testing.PB) {
		foo := 0
		for pb.Next() {
			wg.Add(1)
			for i := 0; i < localWork; i++ {
				foo *= 2
				foo /= 2
			}
			wg.Done()
		}
		_ = foo
	})
}

func BenchmarkWaitGroupAddDone(b *testing.B) {
	benchmarkWaitGroupAddDone(b, 0)
}

func BenchmarkWaitGroupAddDoneWork(b *testing.B) {
	benchmarkWaitGroupAddDone(b, 100)
}

func benchmarkWaitGroupWait(b *testing.B, localWork int) {
	var wg WaitGroup
	b.RunParallel(func(pb *testing.PB) {
		foo := 0
		for pb.Next() {
			wg.Wait()
			for i := 0; i < localWork; i++ {
				foo *= 2
				foo /= 2
			}
		}
		_ = foo
	})
}

func BenchmarkWaitGroupWait(b *testing.B) {
	benchmarkWaitGroupWait(b, 0)
}

func BenchmarkWaitGroupWaitWork(b *testing.B) {
	benchmarkWaitGroupWait(b, 100)
}

func BenchmarkWaitGroupActuallyWait(b *testing.B) {
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			var wg WaitGroup
			wg.Add(1)
			go func() {
				wg.Done()
			}()
			wg.Wait()
		}
	})
}
