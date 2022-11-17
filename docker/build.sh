#!/bin/bash
FILE=$0
cd "$(dirname "${FILE}" )" || exit

docker build \
-t fpenet-sample \
.