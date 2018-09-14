#!/usr/bin/bash
path=$1
echo $path
if [-z path ]
then
	echo "Must Specify Path to Data"
else
	 tar -zxvf $path/* 
fi	
