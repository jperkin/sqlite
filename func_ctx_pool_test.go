// Copyright 2026 The Sqlite Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sqlite // import "modernc.org/sqlite"

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

// These tests pin the pooled *FunctionContext handed to user callbacks
// without dereferencing it, so they hold even when the context they catch is
// stale: two functions evaluated in one statement must receive two different
// sqlite3_context values, and every callback's tls must be the invoking
// connection's. A pooled object that kept a previous invocation's context
// fails both. The other two tests run the pool concurrently and re-entrantly,
// which the sequential probe in func_ctx_test.go cannot.

var ctxIdent struct {
	sync.Mutex
	conn  *conn
	seenA uintptr
	seenB uintptr
	bad   []string
}

func ctxIdentCheck(name string, ctx *FunctionContext) {
	ctxIdent.Lock()
	defer ctxIdent.Unlock()
	if ctx == nil || ctxIdent.conn == nil || ctx.tls != ctxIdent.conn.tls {
		ctxIdent.bad = append(ctxIdent.bad, name+": tls is not the invoking connection's")
	}
}

var (
	ctxPoolCalls  atomic.Int64
	ctxPoolBad    atomic.Int64
	ctxPoolNested atomic.Int64
	ctxPoolDB     atomic.Pointer[sql.DB]
)

func init() {
	MustRegisterDeterministicScalarFunction("ctxident_a", 1, func(ctx *FunctionContext, args []driver.Value) (driver.Value, error) {
		ctxIdentCheck("ctxident_a", ctx)
		ctxIdent.Lock()
		ctxIdent.seenA = ctx.ctx
		ctxIdent.Unlock()
		return args[0], nil
	})
	MustRegisterDeterministicScalarFunction("ctxident_b", 1, func(ctx *FunctionContext, args []driver.Value) (driver.Value, error) {
		ctxIdentCheck("ctxident_b", ctx)
		ctxIdent.Lock()
		ctxIdent.seenB = ctx.ctx
		ctxIdent.Unlock()
		return args[0], nil
	})
	MustRegisterDeterministicScalarFunction("ctxpool_probe", 1, func(ctx *FunctionContext, args []driver.Value) (driver.Value, error) {
		ctxPoolCalls.Add(1)
		if ctx == nil || ctx.tls == nil || ctx.ctx == 0 {
			ctxPoolBad.Add(1)
		}
		return args[0], nil
	})
	// ctxpool_nested runs a statement of its own, so a second trampoline is
	// entered while the first one's pooled object is still checked out.
	MustRegisterScalarFunction("ctxpool_nested", 1, func(ctx *FunctionContext, args []driver.Value) (driver.Value, error) {
		db := ctxPoolDB.Load()
		if db == nil {
			return nil, errors.New("ctxpool_nested: no database registered")
		}
		outer := *ctx
		var v int64
		if err := db.QueryRow(`SELECT ctxpool_probe(?)`, args[0]).Scan(&v); err != nil {
			return nil, err
		}
		if ctx.tls != outer.tls || ctx.ctx != outer.ctx {
			ctxPoolBad.Add(1) // the outer context must survive the nested call untouched
		}
		ctxPoolNested.Add(1)
		return v, nil
	})
}

// TestFunctionContextIdentity checks that the context handed to a callback is
// the one of its own invocation: two different functions in one statement see
// two different sqlite3_context values, and on each of two connections held
// at the same time (so that database/sql cannot hand out one physical
// connection twice) the tls is that connection's own.
func TestFunctionContextIdentity(t *testing.T) {
	db, err := sql.Open(driverName, "file::memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	var conns [2]*sql.Conn
	var raws [2]*conn
	for i := range conns {
		c, err := db.Conn(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		if err := c.Raw(func(dc any) error {
			raws[i] = dc.(*conn)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
		conns[i] = c
	}
	if raws[0] == raws[1] || raws[0].tls == raws[1].tls {
		t.Fatal("the two sql.Conn share one driver connection")
	}

	for i, c := range conns {
		ctxIdent.Lock()
		ctxIdent.conn = raws[i]
		ctxIdent.Unlock()
		var a, b int64
		if err := c.QueryRowContext(context.Background(), `SELECT ctxident_a(1), ctxident_b(2)`).Scan(&a, &b); err != nil {
			t.Fatal(err)
		}
		if a != 1 || b != 2 {
			t.Fatalf("connection %d: got %d, %d", i, a, b)
		}
		ctxIdent.Lock()
		if ctxIdent.seenA == ctxIdent.seenB {
			ctxIdent.bad = append(ctxIdent.bad, "both functions saw the same sqlite3_context")
		}
		ctxIdent.Unlock()
	}

	ctxIdent.Lock()
	defer ctxIdent.Unlock()
	if len(ctxIdent.bad) != 0 {
		t.Fatal(ctxIdent.bad)
	}
}

// TestFunctionContextConcurrent runs the same function on eight connections
// held at the same time, each on its own goroutine, so pooled objects are
// acquired and released concurrently. Run under -race to check the pool's
// synchronization.
func TestFunctionContextConcurrent(t *testing.T) {
	db, err := sql.Open(driverName, "file::memory:?cache=shared")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	db.SetMaxOpenConns(16)

	if _, err := db.Exec(`CREATE TABLE t (a INTEGER)`); err != nil {
		t.Fatal(err)
	}
	for i := int64(1); i <= 50; i++ {
		if _, err := db.Exec(`INSERT INTO t VALUES (?)`, i); err != nil {
			t.Fatal(err)
		}
	}
	ctxPoolCalls.Store(0)
	ctxPoolBad.Store(0)

	const goroutines, rounds = 8, 20
	var wg, holding sync.WaitGroup
	holding.Add(goroutines)
	errs := make(chan error, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := db.Conn(context.Background())
			holding.Done()
			if err != nil {
				errs <- err
				return
			}
			defer c.Close()
			// Every goroutine holds its own connection before any of them
			// starts, so the eight run concurrently rather than in turn
			holding.Wait()
			for r := 0; r < rounds; r++ {
				var s int64
				if err := c.QueryRowContext(context.Background(), `SELECT sum(ctxpool_probe(a)) FROM t`).Scan(&s); err != nil {
					errs <- err
					return
				}
				if s != 1275 {
					errs <- errors.New("sum is not 1275")
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if calls, bad := ctxPoolCalls.Load(), ctxPoolBad.Load(); calls != goroutines*rounds*50 || bad != 0 {
		t.Fatalf("calls=%d bad=%d", calls, bad)
	}
}

// TestFunctionContextNested calls a function that runs a statement invoking
// another function, so a second pooled object is acquired before the first
// is released, and checks the outer context is unchanged afterwards.
func TestFunctionContextNested(t *testing.T) {
	db, err := sql.Open(driverName, "file::memory:?cache=shared")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	db.SetMaxOpenConns(4)
	ctxPoolDB.Store(db)
	defer ctxPoolDB.Store(nil)
	ctxPoolBad.Store(0)
	ctxPoolNested.Store(0)

	var v int64
	for i := int64(1); i <= 100; i++ {
		if err := db.QueryRow(`SELECT ctxpool_nested(?)`, i).Scan(&v); err != nil {
			t.Fatal(err)
		}
		if v != i {
			t.Fatalf("got %d, want %d", v, i)
		}
	}
	if nested, bad := ctxPoolNested.Load(), ctxPoolBad.Load(); nested != 100 || bad != 0 {
		t.Fatalf("nested=%d bad=%d", nested, bad)
	}
}
