#!/bin/bash

problem="mosaic"
grader_name="grader"

g++ -std=gnu++17 -Wall -O2 -pipe -static -g -o "${problem}" "${grader_name}.cpp" "${problem}.cpp"