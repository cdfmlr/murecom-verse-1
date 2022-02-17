package ncmapi

import (
	"errors"
	"fmt"
	"time"
)

// Config: Retries
var (
	NewClientRetryAfterSeconds = 1
	NewClientMaxRetryTimes     = 3
	ClientIsomers              = 1 // 客户端同配置体异示例体
)

// ClientConfig is configs for a new client
type ClientConfig struct {
	Phone       string
	PasswordMD5 string
	Server      string
}

// generator: next client
var nextClient chan *client

// client 的对象池
var pool []*client

// pool.New
func poolInit(configs []ClientConfig) {
	for i := 0; i < ClientIsomers; i++ {
		for _, cfg := range configs {
			c, err := newClient(cfg.Phone, cfg.PasswordMD5, cfg.Server)
			if err == nil { // success
				pool = append(pool, c)
			}
		}
		time.Sleep(200 * time.Millisecond)
	}
	logger.Info(fmt.Sprintf("Pool size=%d", len(pool)))
}

// Init client pools with given clients.
// Offer a Logger to customize logger, while customLogger=nil means using
// default logger.
func Init(configs []ClientConfig, customLogger Logger) error {
	if configs == nil || len(configs) == 0 {
		return errors.New("no clients to use")
	}

	// setup logger
	if customLogger != nil {
		logger = customLogger
	} else {
		logger = newDefaultLogger()
	}

	poolInit(configs)

	// nextClient gen
	nextClient = make(chan *client, len(configs))
	go func() {
		for {
			for _, c := range pool {
				nextClient <- c
			}
		}
	}()

	return nil
}

// region interface

// GetClient get you a Client from the client pool.
// Do not forget call Done(client) to put client back when works are finished.
func GetClient() Client {
	select {
	case c := <-nextClient:
		return c
	case <-time.After(time.Second * 3):
		logger.Error("Pool: no client to use.")
		return nil
	}
}

// Deprecated: Done put c back to the pool
func Done(c Client) {
	// Do nothing.
}

// endregion interface

// TODO: 这个版本随便改的，测试过不了，先不管。
