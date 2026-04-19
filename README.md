# ASPP coursework 2

Please see Learn for submission deadlines.

Remember what you submit must be your own work. Anything not your own
work which you have include should be correctly referenced and
cited. You must not share this assessment's source code nor your
solutions. 

Academic integrity is an underlying principle of research and academic
practice. All submitted work is expected to be your own. AI tools
(e.g., ELM) should not be used for this assessment.

Please see further information and guidance from the School of
Informatics:
<https://informatics.ed.ac.uk/taught-students/all-students/your-studies/academic-misconduct>

## Summary

Starting from a simple MPI-parallelised, CPU-only version of the code
from coursework 1, adapt this to use multiple GPUs on a single EIDF
node (i.e. up to 8 devices). You must choose (with reasons) *one*
programming model from OpenMP, SYCL or CUDA. You should perform a
performance analysis of your code and describe this in a report.

As part of marking, your code will be compiled and run automatically.
It is therefore essential you pay careful attention to the
instructions for submission; if you do not a mark of zero is possible.


## Problem

Here we are solving the wave equation in 3D, using the simplest
method: an explicit, second-order in both space and time finite
difference code. The model has a variable speed of sound and a simple
boundary damping to reduce reflections in the x and y directions.

When run, the code will can produce two (one for each implementation)
output files in HDF5 format which can be read by VTK, or tools based
on it such as ParaView, and visualised. It will also produce timing
and performance information to standard output.

## Set up

Clone this repository on the EIDF (please see Learn for the path),
remembering to use your PVC directory.

Configure with CMake. You MUST specify which GPU programming model you
are using by setting `AWAVE_MODE` to one of `OpenMP`, `CUDA`, or
`SYCL`. 

```
cmake -S src -B build-$yourchoice -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAWAVE_MODE=$yourchoice
```

Note: for marking, we will use a build mode of `Release`, which is the
default, but you may wish to include basic debug info for testing
while doing your development.


Compile
```
cmake --build build-$yourchoice
```

The unmodified code will run on the ASPP VM (as it doesn't use the
GPUs!) but once you start work, you will need to use the compute
nodes. You will need to write your own job descriptions (the first
coursework and the multi-GPU practicals should provide a starting
point).

The application first runs a CPU reference simulation then your
selected GPU implementation, comparing the results of the latter two
to the former with some tolerance. It then reports on timing
information.

Usage:
```
Usage: awave [flags] [output_base]
output_base defaults to 'test'

Flag               | Default  | Description
-------------------|----------|-------------
-mpi int,int,int   |    1,1,1 | Layout of the grid of MPI processes
-shape int,int,int | 32,32,32 | Shape of the domain to simulated=
-dx float          |     10.0 | Space step in metres
-dt float          |    0.002 | Time step in seconds
-nsteps int        |      100 | Number of time steps
-out_period int    |       10 | Number of time steps between outputs
-restart path      |  nullopt | Restart from checkpoint (ignores parameters above)
-io (0|1)          |    false | Actually write output
-cpu_serial        |    false | Force CPU run to be single MPI process
-json path         |  nullopt | File to write JSON output to
```

For your purposes changing the the process and domain shapes should be
sufficient. Note that during marking we will run at least one
checkpoint / restart case which may have different speed of sound
and/or damping fields from the simple parameterised cases.

## Requirements

You need to prepare a brief report and adapt the code (only submitting
one file from `wave_cuda.cu`, `wave_omp.cpp`, or `wave_sycl.cpp`) to
use a single node with the best performance you can.

### Report

This should be in PDF format, with a maximum of five pages and a font
size of at least 10pts. Please include the sections
below. Introduction and conclusions are not required. Remember to
properly cite any sources. [Suggested length allocations in brackets.]

1. Design and process.

Briefly explain the approach you have taken to developing your code,
the key choices made and the reasons for these choices. One key point
is your choice of programming model for this problem.

You may wish to include profiling/timing results from interim versions
of your code. You should address both single-device performance as
well as steps relating to message passing. You may draw on evidence
from your first coursework. [2 pages]

2. Results

Show performance results (remembering that incorrect answers have zero
performance) for a range of problem sizes, number of devices, and
numbers of MPI processes.
 - You should use *both* A100 80GB and H100 nodes with a device count
   up to (and including) 8
 - Problem sizes should range from 256 to 2500 (inclusive)
 - The number of MPI processes is up to you
You must include appropriate figures and discuss your results. You
should consider how close you have got to the theoretical performance
limits of the node. [2 pages]

3. Further work

Estimate how your submitted implementation would perform (for large
enough problems) if scaled out across multiple (homogeneous) nodes
connected with a high-performance interconnect. Discuss how your code,
and the wider application, might need to be changed to improve this
scaling to be efficient on hundreds of nodes (thousands of GPUs).

For this section, you may negelect any consideration of IO.

### Code

Adapt the code to use the resources of a node efficiently with your
chosen programming model. You must ensure all device work and
communications are finished inside your `run` function.

- For OpenMP target offload use the file `wave_omp.cpp`
- For CUDA use the file `wave_cuda.cu`
- For SYCL use the file `wave_sycl.cpp`

Your code will be assessed for correctness, performance, and
clarity.

You will be submitting only *one* of the files listed above, so please
localise your changes to that file (you can always check with `git
status`) as we will keep the rest of the application as is supplied in
the repo. The supplied code has further advice in the source files.

*Clarity*: your modifications will be marked for good software
engineering practice. Recall that you are explaining to an experienced
programmer how the node's hardware will coordinate to solve a problem.

*Correctness*: the driver program compares the GPU results to those
from the (unmodified) CPU version. We will compile and run this
against a range of problem shapes, thus you should also in your
testing. Ensure the program correctly manages any resources it uses.

*Performance*: we will use several problems of different sizes, taking
the best performance run in each case. Sizes will be between 256 to
2500 inclusive along each dimension.

## Submission

Please create a gzipped tar file which unpacks to a directory named
for your exam number containing your chosen implementation file and a
PDF report, e.g. for SYCL:
```
examno/wave_sycl.cpp
examno/<sensible report name>.pdf
```

We have provided a script that will create this for you: 
```
./make_submission.sh -h
Usage: ./make_submission.sh (-h |-e <exam number> -r <path/to/report>)
Create a submission-ready tarball as aspp-cw1-<exam number>.tar.gz
  -h print this message and exit without doing anything
  -e your exam number, the one that starts with a 'B' and has six numbers after
  -r your report in PDF format please

Both -e and -r options are required to produce a tarball.
```

## Marking rubric

Marks for the report will be awarded according to the following scheme:

   80% An excellent report, fully addressing all points above, and
       presenting convincing evidence/results. The report is written
       and presented to a professional standard.

70-80% A very good report, addressing all points to a good standard,
	   and presenting evidence and results that supports the claims
	   made, irrelevant text is minimal. The report is well written
	   and presented.

60-70% A good report, addressing all the points with only minor
	   omissions or confusion. Some irrelevancies, but not detracting
	   from the report generally. Evidence and results support the
	   points made, but perhaps not to the extent claimed. Writing and
	   presentation generally good.

50-60% An OK report, addressing all of the points required at least
	   partially but perhaps showing some faulty reasoning, minor
	   confusions or longer passages of irrelevant
	   material. Evidence/results are present but not convincing or
	   contains minor inaccuracies. Writing and presentation are
	   generally satisfactory but with some brief lapses.

40-50% A poor report with either significant omissions, failures in
       reasoning or that contains significant irrelevant
       material. Evidence or results have significant omissions or do
       not support the claims made. Writing and presentation have
       significant flaws.

  <40% A weak effort which is incomplete or incomprehensible

Marks for the codes will be awarded according to the following scheme:

   80% Correct, very well-designed code which, clearly and succintly,
	   explains how the CPUs and GPUs coordinate to get performance,
	   very good performance, likely using one or more advanced
	   techniques.

70-80% Correct well-designed code with clear explanation of CPU/GPU
	   use, and good performance.

60-70% Correct code, with generally clear GPU/CPU use and good
	   performance. Likely doing the basics well or attempting the
	   more advanced techniques with minor problems.

50-60% Code with minor errors, some oversights in explaining GPU/CPU
	   use and some design and/or performance issues. May fail to
	   properly consider an important performance issue.

40-50% Code with more serious errors, poor design and/or poor
       performance. No serious attempt to make clear the how
       GPU/CPU coordinate.

  <40% Badly broken or incomplete code and/or very bad
       design. Description/comments missing or incomprehensible.
