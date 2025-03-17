#!/bin/bash
RED='\033[0;31m'
NC='\033[0m' # No Color
printf "I ${RED}love${NC} Stack Overflow\n"

for i in */; do
 extension=${i%/}
 if [ -f "${extension}/${extension}_results.json" ]; then
  echo "$extension done" 
 else
  echo -e "${RED}$extension${NC} not done" 
 fi
done
