package neteasecloudmusic

import (
	"fmt"
	"log"
)

type Logger interface {
	Info(s string)
	Warn(s string)
	Error(s string)
}

// logger for neteasecloudmusic package
var logger Logger = newDefaultLogger()

// region defaultLogger

type defaultLogger log.Logger

func newDefaultLogger() Logger {
	dl := defaultLogger(*log.Default())
	return &dl
}

// log prints a default log like this:
//    [Level] neteasecloudmusic: something happened
func (l *defaultLogger) log(level string, s string) {
	(*log.Logger)(l).Printf("[%s] neteasecloudmusic: %s\n", level, s)
}

func (l *defaultLogger) Info(s string) {
	l.log("Info", s)
}

func (l *defaultLogger) Warn(s string) {
	l.log("Warn", s)
}

func (l *defaultLogger) Error(s string) {
	l.log("Error", s)
}

// endregion defaultLogger

// region debug

const innerDebug = false

func debug(s ...interface{}) {
	if innerDebug {
		logger.Info("[debug] " + fmt.Sprint(s...))
	}
}

// endregion debug
