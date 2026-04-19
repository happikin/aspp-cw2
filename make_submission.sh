#!/bin/bash
set -e

RUN_DIR=$PWD
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

usage="Usage: ./make_submission.sh (-h |-e <exam number> -r <path/to/report> -m <mode>)
Create a submission-ready tarball as aspp-cw2-<exam number>.tar.gz
  -h print this message and exit without doing anything
  -e your exam number, the one that starts with a 'B' and has six numbers after
  -r your report in PDF format please
  -m programming model, one of CUDA, OpenMP, SYCL

All of -e, -r, and -m options are required to produce a tarball.
"

while [[ $# -gt 0 ]]; do
    case $1 in
    -h)
	echo "$usage"
	exit 0
	shift;;
    -e)
	shift
	if [[ $# -lt 1 ]]; then
	    echo "Missing value for -e"
	fi
	examno=$1
	shift;;
    -r)
	shift
	if [[ $# -lt 1 ]]; then
	    err "Missing value for -e"
	fi
	report=$1
	shift;;
    -m)
	shift
	if [[ $# -lt 1 ]]; then
	    err "Missing value for -m"
	fi
	mode=$1
	shift;;
    -*)
	echo "unknown option: $1"
	exit 1;;
    *)
	echo "unexpected argument: $1"
	exit 1;;
    esac
done

if [ -z "$examno" ]; then
    echo "Missing -e <exam number>"
    echo "$usage"
    exit 1
fi

if ! [[ "$examno" =~ ^B[0-9]{6}$ ]]; then
    echo "Invalid exam number: $examno"
    exit 1
else
    echo "Exam number: $examno"
fi

if [ -z "$report" ]; then
    echo "Missing -r <report>"
    echo "$usage"
    exit 1
fi

if [ -f "$report" ]; then
    ft=$(file -b $report)
    if ! [[ "$ft" =~ ^PDF ]]; then
	echo "Report does not appear to be a PDF: $(file $report)"
	exit 1
    fi
else
    echo "Report is not a file: $report"
    exit 1
fi

echo "Report is a PDF"

if [ -z "$mode" ]; then
    echo "Missing -m <mode>"
    echo "$usage"
    exit 1
fi

case $mode in
    OpenMP)
	impl=wave_omp.cpp;;
    CUDA)
	impl=wave_cuda.cu;;
    SYCL)
	impl=wave_sycl.cpp;;
    *)
	echo "mode not one of OpenMP, CUDA, SYCL"
	exit 1;;
esac

tmpdir=$(mktemp -d)
echo "Making temporary directory"
mkdir $tmpdir/$examno

echo "Copy report"
cd "$RUN_DIR"
cp "$report" $tmpdir/$examno/$examno.pdf

echo "Copy source files"
cd "$SCRIPT_DIR/src"
cp $impl $tmpdir/$examno/

echo "Create tarball"
tarball=aspp-cw2-$examno.tar.gz
cd $tmpdir
tar -czf $tarball $examno

cd "$RUN_DIR"
cp $tmpdir/$tarball ./

echo "Clean up tmpdir"
rm -rf $tmpdir
