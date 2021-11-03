// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package context defines the Context type, which carries deadlines,
// cancellation signals, and other request-scoped values across API boundaries
// and between processes.
//
// Incoming requests to a server should create a Context, and outgoing
// calls to servers should accept a Context. The chain of function
// calls between them must propagate the Context, optionally replacing
// it with a derived Context created using WithCancel, WithDeadline,
// WithTimeout, or WithValue. When a Context is canceled, all
// Contexts derived from it are also canceled.
//
// The WithCancel, WithDeadline, and WithTimeout functions take a
// Context (the parent) and return a derived Context (the child) and a
// CancelFunc. Calling the CancelFunc cancels the child and its
// children, removes the parent's reference to the child, and stops
// any associated timers. Failing to call the CancelFunc leaks the
// child and its children until the parent is canceled or the timer
// fires. The go vet tool checks that CancelFuncs are used on all
// control-flow paths.
//
// Programs that use Contexts should follow these rules to keep interfaces
// consistent across packages and enable static analysis tools to check context
// propagation:
//
// Do not store Contexts inside a struct type; instead, pass a Context
// explicitly to each function that needs it. The Context should be the first
// parameter, typically named ctx:
//
// 	func DoSomething(ctx context.Context, arg Arg) error {
// 		// ... use ctx ...
// 	}
//
// Do not pass a nil Context, even if a function permits it. Pass context.TODO
// if you are unsure about which Context to use.
//
// Use context Values only for request-scoped data that transits processes and
// APIs, not for passing optional parameters to functions.
//
// The same Context may be passed to functions running in different goroutines;
// Contexts are safe for simultaneous use by multiple goroutines.
//
// See https://blog.golang.org/context for example code for a server that uses
// Contexts.
package context

import (
	"errors"
	"internal/reflectlite"
	"sync"
	"sync/atomic"
	"time"
)

// Context 用以在多 API 间传递 deadline、cancelation 信号和请求的键值对。
// Context 中的方法能够安全的被多个 goroutine 并发调用。
type Context interface {
	// 如果 Context 被设置了超时，Deadline 将会返回超时时限。
	Deadline() (deadline time.Time, ok bool)
	// Done 返回一个只读 channel，该 channel 在 Context 被取消或者超时时关闭
	Done() <-chan struct{}
	// Err 返回 Context 结束时的出错信息
	Err() error
	// Value 返回关联到相关 Key 上的值，或者 nil.
	Value(key interface{}) interface{}
}

// Canceled is the error returned by Context.Err when the context is canceled.
var Canceled = errors.New("context canceled")

// DeadlineExceeded is the error returned by Context.Err when the context's
// deadline passes.
var DeadlineExceeded error = deadlineExceededError{}

type deadlineExceededError struct{}

func (deadlineExceededError) Error() string   { return "context deadline exceeded" }
func (deadlineExceededError) Timeout() bool   { return true }
func (deadlineExceededError) Temporary() bool { return true }

// An emptyCtx is never canceled, has no values, and has no deadline. It is not
// struct{}, since vars of this type must have distinct addresses.
type emptyCtx int

func (*emptyCtx) Deadline() (deadline time.Time, ok bool) {
	return
}

func (*emptyCtx) Done() <-chan struct{} {
	return nil	// 返回 nil，从语法上说是空实现，从语义上说是该 Context 永远不会被关闭。
}

func (*emptyCtx) Err() error {
	return nil
}

func (*emptyCtx) Value(key interface{}) interface{} {
	return nil
}

func (e *emptyCtx) String() string {
	switch e {
	case background:
		return "context.Background"
	case todo:
		return "context.TODO"
	}
	return "unknown empty Context"
}

var (
	background = new(emptyCtx)
	todo       = new(emptyCtx)
)

// Background 返回一个空 Context。它不能被取消，没有时限，没有附加键值。Background 通常用在
// main函数、init 函数、test 入口，作为某个耗时过程的根 Context。
func Background() Context {
	return background
}

// TODO returns a non-nil, empty Context. Code should use context.TODO when
// it's unclear which Context to use or it is not yet available (because the
// surrounding function has not yet been extended to accept a Context
// parameter).
func TODO() Context {
	return todo
}

// 调用 CancelFunc 取消对应 Context.
type CancelFunc func()

// WithCancel 返回一份父 Context 的拷贝，和一个 cancel 函数。当父 Context 被关闭或者
// 此 cancel 函数被调用时，该 Context 的 Done Channel 会立即被关闭.
func WithCancel(parent Context) (ctx Context, cancel CancelFunc) {
	if parent == nil {
		panic("cannot create context from nil parent")
	}
	c := newCancelCtx(parent)
	propagateCancel(parent, &c)
	return &c, func() { c.cancel(true, Canceled) }
}

// newCancelCtx returns an initialized cancelCtx.
func newCancelCtx(parent Context) cancelCtx {
	return cancelCtx{Context: parent}
}

// goroutines counts the number of goroutines ever created; for testing.
var goroutines int32

// -实现：沿着回溯链找到第一个实现了 Done() 方法的实例
// -目的：propagateCancel主要设计目标就是当parent context取消的时候，进行child context的取消
// -模式：
// 1.parent取消的时候通知child进行cancel取消
// 2.parent取消的时候调用child的层层递归取消

func propagateCancel(parent Context, child canceler) {
	done := parent.Done()
	if done == nil {
		return 	// 父节点不可取消
	}

	select {
	case <-done:
		// 父节点已经取消
		child.cancel(false, parent.Err())
		return
	default:
	}

	if p, ok := parentCancelCtx(parent); ok {	// 找到第一个 是cancelCtx 实例
		p.mu.Lock()
		if p.err != nil {
			// 父节点已经被取消
			child.cancel(false, p.err)
		} else {
			if p.children == nil {
				p.children = make(map[canceler]struct{})	// 惰式创建
			}
			p.children[child] = struct{}{}
		}
		p.mu.Unlock()
	} else {	// 找到一个非 cancelCtx 实例
		atomic.AddInt32(&goroutines, +1)
		go func() {
			select {
			case <-parent.Done():
				child.cancel(false, parent.Err())
			case <-child.Done():
			}
		}()
	}
}

// &cancelCtxKey is the key that a cancelCtx returns itself for.
var cancelCtxKey int

// parentCancelCtx 返回 parent 的第一个祖先 cancelCtx 节点
func parentCancelCtx(parent Context) (*cancelCtx, bool) {
	done := parent.Done()	// 调用回溯链中第一个实现了 Done() 的实例(第三方Context类/cancelCtx)
	// 注：
	// 1.在cancel的value==nil时，赋值closedchan
	// 2.context.TODO或者context.Background，因为cancelctx 在调用Done（）时会make，不为nil
	if done == closedchan || done == nil {
		return nil, false
	}
	p, ok := parent.Value(&cancelCtxKey).(*cancelCtx)	// 回溯链中第一个 cancelCtx 实例
	if !ok {
		return nil, false
	}
	// 说明回溯链中第一个实现 Done() 的实例不是 cancelCtx 的实例
	pdone, _ := p.done.Load().(chan struct{})
	if pdone != done {
		return nil, false
	}
	return p, true
}

// removeChild removes a context from its parent.
func removeChild(parent Context, child canceler) {
	p, ok := parentCancelCtx(parent)
	if !ok {
		return
	}
	p.mu.Lock()
	if p.children != nil {
		delete(p.children, child)
	}
	p.mu.Unlock()
}

// Context 树，本质上可以细化为 canceler （*cancelCtx 和 *timerCtx）树，
// 因为在级联取消时只需找到子树中所有的 canceler ，
// 因此在实现时只需在树中保存所有 canceler 的关系即可（跳过 valueCtx），简单且高效。
type canceler interface {
	cancel(removeFromParent bool, err error)
	Done() <-chan struct{}
}

// closedchan is a reusable closed channel.
var closedchan = make(chan struct{})

func init() {
	close(closedchan)
}


type cancelCtx struct {
	Context

	mu       sync.Mutex            // 保证下面三个字段的互斥访问
	done     atomic.Value          // 惰式初始化，被第一个 cancel() 调用所关闭
	children map[canceler]struct{} // 被第一个 cancel() 调用置 nil
	err      error                 // 被第一个 cancel() 调用置非 nil
}

func (c *cancelCtx) Value(key interface{}) interface{} {
	//go 用了一个特殊的 key：cancelCtxKey，遇到该 key 时，cancelCtx 会返回自身。
	if key == &cancelCtxKey {
		return c
	}
	return c.Context.Value(key)
}

//  Done操作返回当前的一个chan 用于通知goroutine退出
func (c *cancelCtx) Done() <-chan struct{} {
	d := c.done.Load()
	if d != nil {
		return d.(chan struct{})
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	d = c.done.Load()
	if d == nil {
		d = make(chan struct{})
		c.done.Store(d)
	}
	return d.(chan struct{})
}

func (c *cancelCtx) Err() error {
	c.mu.Lock()
	err := c.err
	c.mu.Unlock()
	return err
}

type stringer interface {
	String() string
}

func contextName(c Context) string {
	if s, ok := c.(stringer); ok {
		return s.String()
	}
	return reflectlite.TypeOf(c).String()
}

func (c *cancelCtx) String() string {
	return contextName(c.Context) + ".WithCancel"
}

func (c *cancelCtx) cancel(removeFromParent bool, err error) {
	// 需要给定取消的理由，Canceled or DeadlineExceeded
	if err == nil {
		panic("context: internal error: missing cancel error")
	}
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return // 已经被其他 goroutine 取消
	}
	// 记下错误，并关闭 done
	c.err = err
	// 如果c.done等于nil，c.done 会被赋值为closedchan
	d, _ := c.done.Load().(chan struct{})
	if d == nil {
		c.done.Store(closedchan)
	} else {
		close(d)
	}
	// 级联取消
	for child := range c.children {
		// NOTE: 持有父 Context 的同时获取了子 Context 的锁
		child.cancel(false, err)
	}
	c.children = nil
	c.mu.Unlock()

	// 子树根需要摘除，子树中其他节点则不再需要

	// 是否需要从parent context中移除,如果是当前context的取消操作，则需要进行该操作
	// 否则，则上层context会主动进行child的移除工作
	if removeFromParent {
		removeChild(c.Context, c)
	}
}

// 1.设置超时取消是在 context.WithDeadline() 中完成的。
// 2.如果祖先节点时限早于本节点，只需返回一个 cancelCtx 即可，
// 因为祖先节点到点后在级联取消时会将其取消。
func WithDeadline(parent Context, d time.Time) (Context, CancelFunc) {
	if parent == nil {
		panic("cannot create context from nil parent")
	}
	if cur, ok := parent.Deadline(); ok && cur.Before(d) {
		// The current deadline is already sooner than the new one.
		return WithCancel(parent)
	}
	c := &timerCtx{
		cancelCtx: newCancelCtx(parent),
		deadline:  d,
	}
	propagateCancel(parent, c)	// 构建 Context 取消树，注意传入的是 c 而非 c.cancelCtx
	dur := time.Until(d)		// 测试时限是否设的太近以至于已经结束了
	if dur <= 0 {
		// 已经过期
		c.cancel(true, DeadlineExceeded) // deadline has already passed
		return c, func() { c.cancel(false, Canceled) }
	}
	// 设置超时取消
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.err == nil {
		c.timer = time.AfterFunc(dur, func() {
			// 构建一个timer定时器，到期后自动调用cancel取消
			c.cancel(true, DeadlineExceeded)
		})
	}
	// 返回取消函数
	return c, func() { c.cancel(true, Canceled) }
}

// timerCtx 在嵌入 cancelCtx 的基础上增加了一个计时器 timer，
// 根据用户设置的时限，到点取消。
type timerCtx struct {
	cancelCtx
	timer *time.Timer // Under cancelCtx.mu.

	deadline time.Time
}

func (c *timerCtx) Deadline() (deadline time.Time, ok bool) {
	return c.deadline, true
}

func (c *timerCtx) String() string {
	return contextName(c.cancelCtx.Context) + ".WithDeadline(" +
		c.deadline.String() + " [" +
		time.Until(c.deadline).String() + "])"
}

func (c *timerCtx) cancel(removeFromParent bool, err error) {
	// 级联取消子树中所有 Context
	c.cancelCtx.cancel(false, err)
	if removeFromParent {
		// 单独调用以摘除此节点，因为是摘除 c，而非 c.cancelCtx
		removeChild(c.cancelCtx.Context, c)
	}
	// 关闭计时器
	c.mu.Lock()
	if c.timer != nil {
		c.timer.Stop()
		c.timer = nil
	}
	c.mu.Unlock()
}

// WithTimeout 返回一份父 Context 的拷贝，和一个 cancel 函数。当父 Context 被关闭、
// cancel 函数被调用或者设定时限到达时，该 Context 的 Done Channel 会立即关闭。在 cancel 函数
// 被调用时，如果其内部 timer 仍在运行，将会被停掉。
func WithTimeout(parent Context, timeout time.Duration) (Context, CancelFunc) {
	return WithDeadline(parent, time.Now().Add(timeout))
}

// WithValue 返回一个父 Context 的副本，并且附加上给定的键值对.
func WithValue(parent Context, key, val interface{}) Context {
	if parent == nil {
		panic("cannot create context from nil parent")
	}
	if key == nil {
		panic("nil key")
	}
	if !reflectlite.TypeOf(key).Comparable() {
		panic("key is not comparable")
	}
	return &valueCtx{parent, key, val}	// 附加上 kv，并引用父 Context
}


type valueCtx struct {
	Context	// 嵌入，指向父 Context
	key, val interface{}
}

// stringify tries a bit to stringify v, without using fmt, since we don't
// want context depending on the unicode tables. This is only used by
// *valueCtx.String().
func stringify(v interface{}) string {
	switch s := v.(type) {
	case stringer:
		return s.String()
	case string:
		return s
	}
	return "<not Stringer>"
}

func (c *valueCtx) String() string {
	return contextName(c.Context) + ".WithValue(type " +
		reflectlite.TypeOf(c.key).String() +
		", val " + stringify(c.val) + ")"
}

func (c *valueCtx) Value(key interface{}) interface{} {
	if c.key == key {
		return c.val
	}
	return c.Context.Value(key)
}
