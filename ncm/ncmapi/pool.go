package ncmapi

import (
	"errors"
	"strconv"
	"sync"
	"time"
)

// Config: Retries
var (
	NewClientRetryAfterSeconds = 1
	NewClientMaxRetryTimes     = 3
)

// ClientConfig is configs for a new client
type ClientConfig struct {
	Phone       string
	PasswordMD5 string
	Server      string
}

// generator: next config for new client
var nextClientConfig chan ClientConfig

// client 的对象池
var pool *sync.Pool

// pool.New
func poolNew() interface{} {
	debug("poolNew in")
	for i := 0; i < NewClientMaxRetryTimes; i++ {
		debug("poolNew try =", i)
		cfg := <-nextClientConfig // 这个在循环里，换一个号重试：可能配置是错的，不要死磕一个号
		debug("poolNew got cfg:", cfg)
		c, err := newClient(cfg.Phone, cfg.PasswordMD5, cfg.Server)
		if err == nil { // success
			debug("new client:", c)
			return c
		}
		logger.Error("failed to create new client, try again after " + strconv.Itoa(NewClientRetryAfterSeconds) + " seconds...")
		time.Sleep(time.Duration(NewClientRetryAfterSeconds) * time.Second)
	}
	return nil
}

// Init client pools with given clients.
// Offer a Logger to customize logger, while customLogger=nil means using
// default logger.
func Init(clients []ClientConfig, customLogger Logger) error {
	if clients == nil || len(clients) == 0 {
		return errors.New("no clients to use")
	}

	// setup logger
	if customLogger != nil {
		logger = customLogger
	} else {
		logger = newDefaultLogger()
	}

	// nextClientConfig gen
	nextClientConfig = make(chan ClientConfig, len(clients))
	go func() {
		for {
			for _, c := range clients {
				nextClientConfig <- c
			}
		}
	}()

	// init pool
	pool = &sync.Pool{New: poolNew}
	return nil
}

// region interface

// GetClient get you a Client from the client pool.
// Do not forget call Done(client) to put client back when works are finished.
func GetClient() Client {
	debug("Debug: GetClient in")
	g := pool.Get()
	debug("Debug: pool.Get:", g)
	switch c := g.(type) {
	case Client:
		return c
	}
	logger.Error("GetClient return nil: no client to use")
	return nil
}

// Done put c back to the pool
func Done(c Client) {
	switch c := c.(type) {
	case *client:
		pool.Put(c)
	}
	// else: rubbish
}

// endregion interface
