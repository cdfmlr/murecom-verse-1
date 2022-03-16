package ncm

import (
	"fmt"
	"log"
	"os"
)

const callDepth = 3

type Logger struct {
	*log.Logger
}

func (l *Logger) log(calldepth int, level string, a ...interface{}) {
	_ = l.Output(calldepth, "["+level+"] "+fmt.Sprint(a...))
}

func (l *Logger) Info(a ...interface{}) {
	l.log(callDepth, "Info", a...)
}

func (l *Logger) Warn(a ...interface{}) {
	l.log(callDepth, "Warn", a...)
}

func (l *Logger) Error(a ...interface{}) {
	l.log(callDepth, "Error", a...)
}

func (l *Logger) Debug(a ...interface{}) {
	if Debug {
		l.log(callDepth, "Debug", a...)
	}
}

func (l *Logger) Progress(done, all int, a ...interface{}) {
	p := done * 100 / all
	l.log(callDepth, fmt.Sprintf("Progress %v%%", p), a...)
}

var logger Logger

func init() {
	logger = Logger{log.New(os.Stdout, "",
		log.Ldate|log.Ltime|log.Lshortfile)}
}

type ncmapiLogger struct {
	Logger
}

func (n ncmapiLogger) Info(s string) { n.Logger.log(callDepth, "ncmapi Info", s) }

func (n ncmapiLogger) Warn(s string) { n.Logger.log(callDepth, "ncmapi Warn", s) }

func (n ncmapiLogger) Error(s string) { n.Logger.log(callDepth, "ncmapi Error", s) }
