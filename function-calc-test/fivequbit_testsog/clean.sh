#!/bin/bash
find . -name '*out' -exec rm {} \;
find . -name '*svg' -exec rm {} \;
find . -name '*csv' -exec rm {} \;
find . -name '*bin' -exec rm {} \;
find . -name '*results.json' -exec rm {} \;
find . -name '*search.json' -exec rm {} \;
