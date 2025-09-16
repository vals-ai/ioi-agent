#!/bin/sh
set -e

TASK=mosaic
LANGU=c++17

if [ $# != 1 ] ; then
	echo >&2 "Usage $0 <source-file>"
	exit 1
fi

exec ioisubmit --lang=$LANGU $TASK "$1"
