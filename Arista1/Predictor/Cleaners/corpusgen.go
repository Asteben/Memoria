package main

import (
"bufio"
"encoding/csv"
"encoding/json"
"fmt"
"log"
"os"
"sort"
"strconv"
"strings"
"time"
)

type twitData struct {
	Unix 		int64
	CreatedAt	string	`json:"created_at"`
}
type parsedData struct {
	Unix int64
	Quantity int
}

func evalErr( err error ) {
	if err != nil {
		log.Println(err)
	}
}

func readRaw( fileName string ) []twitData {
	file , err := os.Open( fileName )
	evalErr(err)
 	defer file.Close()
	var result []twitData
	scanner := bufio.NewScanner( file )
	for scanner.Scan() {
		line := scanner.Text()
		var obj twitData
		err := json.Unmarshal( json.RawMessage(line) , &obj )
		evalErr(err)
		date , err := time.Parse( time.RubyDate , obj.CreatedAt )
		evalErr(err)
		obj.Unix = date.Unix()
		result = append(result , obj)
	}
 	err = scanner.Err()
	evalErr(err)
	file.Close()
	return result
}
func parseRaw( result []twitData , secondsInterval int64) []parsedData {
	sort.Slice( result , func (i, j int ) bool {
		return result[i].Unix < result[j].Unix
	})
	var resultGrouped []parsedData
	c := 0
	current := ( result[0].Unix - result[0].Unix %secondsInterval )
	max := ( result[ len( result ) -1].Unix + secondsInterval - result[ len( result ) -1].Unix %secondsInterval )
	for i := 0; i < len( result ) && current < ( max + secondsInterval); current += secondsInterval {
 		for ; i < len( result ) && current > result[i].Unix ; i++ {
			c++
			fmt.Println(i)
		}
		temp := parsedData { current , c}
		resultGrouped = append( resultGrouped , temp )
		c = 0
	}
	return resultGrouped
}

func writeCsv( parsedData []parsedData , outFilePath string ) {
	var parsedResult [][]string
	for _, value := range parsedData {
		var temp []string
		temp = append(temp , strconv.FormatInt( value.Unix , 10))
		temp = append(temp , strconv.Itoa( value.Quantity ))
		parsedResult = append( parsedResult , temp )
	}
	fileCsv , err := os.Create( outFilePath )
	evalErr(err)
	defer fileCsv.Close()
	writer := csv.NewWriter( fileCsv )
	defer writer.Flush()
	for _, value := range parsedResult {
		err := writer.Write( value )
		evalErr(err)
	}
}

func main() {
	inFilePath := "dataset/dataset.jsonl"
	outFilePath := "dataset.csv"
	result := readRaw( inFilePath )
	resultGrouped := parseRaw( result , 5)
	writeCsv( resultGrouped , strings.Replace( outFilePath , "dataset.csv", "dataset1h_5.csv", -1) )
	//resultGrouped = parseRaw( result , country , disasterType , 60)
	//writeCsv( resultGrouped , strings.Replace( outFilePath , "dataset.csv", "dataset1h_60.csv", -1) )
	//resultGrouped = parseRaw( result , country , disasterType , 180)
	//writeCsv( resultGrouped , strings.Replace( outFilePath , "dataset.csv", "dataset1h_180.csv" , -1) )
	//resultGrouped = parseRaw( result , country , disasterType , 300)
	//writeCsv( resultGrouped , strings.Replace( outFilePath , "dataset.csv", "dataset1h_300. csv" , -1) )
}