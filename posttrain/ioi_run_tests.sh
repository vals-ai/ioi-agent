#!/usr/bin/env bash
set -u
if [ "$#" -lt 2 ]; then
    echo 'Error: memory_limit and time_limit are required'
    exit 2
fi
g++ -std=gnu++20 -Wall -O2 -pipe -g -o solution -I graders -include bits/stdc++.h graders/grader.cpp solution.cpp || { echo 'Error: Failed to compile solution'; exit 3; }
g++ -std=gnu++20 -Wall -O2 -pipe -g -o checkr -I checker checker/checker.cpp || { echo 'Error: Failed to compile checker'; exit 4; }
passed=''
status=0
for input in tests/*.in; do
    name=$(basename "$input" .in)
    if ! (ulimit -v "$1"; timeout "$2" ./solution < "$input" > output.txt); then
        echo "Error: Solution execution failed for test case $name"
        status=1
        continue
    fi
    if result=$(./checkr "$input" "tests/$name.out" output.txt 2>&1); then
        if [ "$(printf '%s\n' "$result" | tail -n 1)" = 1 ]; then
            passed="$passed $name"
        else
            status=1
        fi
    else
        echo "Error: Checker execution failed for test case $name"
        status=1
    fi
done
if [ "$status" -eq 0 ]; then echo 'All tests passed successfully!'; fi
echo "Passed tests:$passed"
exit "$status"
