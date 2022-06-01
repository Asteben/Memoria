package main

import (
	"fmt"
	"log"
	"time"
)

func evalErr( err error ) {
	if err != nil {
		log.Println(err)
	}
}

type twitData struct {
	CreatedAt string `json :"created_at"`
	Unix int64
}

func main() {
    t, err := time.Parse(time.RubyDate, "Thu Mar 19 19:52:18 +0000 2020")
	evalErr(err)
    fmt.Println(t)
}