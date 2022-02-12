package ncmapi

import (
	"fmt"
	"log"
	"testing"
)

type testLogger struct {
	log.Logger
}

func (t *testLogger) Info(s string) {
	t.Println("testLogger.Info: ", s)
}

func (t *testLogger) Warn(s string) {
	t.Println("testLogger.Warn: ", s)
}

func (t *testLogger) Error(s string) {
	t.Println("testLogger.Error: ", s)
}

func TestLogger(t *testing.T) {
	type args struct {
		clients      []ClientConfig
		customLogger Logger
	}
	tests := []struct {
		name string
		args args
	}{
		{"defaultLogger", args{
			clients:      []ClientConfig{testClientConfig},
			customLogger: nil,
		}},
		{"customLogger", args{
			clients:      []ClientConfig{testClientConfig},
			customLogger: &testLogger{*log.Default()},
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Init(tt.args.clients, tt.args.customLogger)
			if err != nil {
				t.Error(err)
			}
			logger.Info("info test" + fmt.Sprint(tt.args))
			logger.Warn("warn test" + fmt.Sprint(tt.args))
			logger.Error("error test" + fmt.Sprint(tt.args))
		})
	}
}
