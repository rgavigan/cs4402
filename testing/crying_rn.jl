###############
## FUNCTIONS ##
###############

######################################
## Julia is a lightly-type language ##
######################################

function f(x,y)
           x + y
       end

f(2,3)


fib(n::Integer) = n ≤ 2 ? one(n) : fib(n-1) + fib(n-2)

fib(10)

#fib(3.14)

#########################################################
## Julia function arguments follow  "pass-by-sharing"  ##
#########################################################

function g(x,y)
           x = x + y
       end

a = [1,2]
b = [3,4]

g(a,b)

a

function h(u,v)
         u[1] = v
	 end
h(a,3)
print(a)

a


###############################
## Using anonymous Functions ##
###############################

map(x -> x^2 + 2x - 1, [1, 3, -1])

map(x->begin
           if x < 0 && iseven(x)
               return 0
           elseif x == 0
               return 1
           else
               return x
           end
       end,
    [-2, 0, 2])


########################################
## Julia has many syntactic sugars    ##
########################################

(a,b,c) = 1:3

a
b
c

a, b... = "hello"

a
b


(sqrt ∘ +)(3, 6)


1:10 |> sum |> sqrt


##########################################
## Dot Syntax for Vectorizing Functions ##
##########################################


A = [1.0, 2.0, 3.0]

sin.(A)

(x -> x+ 1).(A)


#############
## METHODS ##
#############

###########################################################
## Number is an abstract supertype for all number types. ##
###########################################################

f(x::Float64, y::Float64) = 2x + y


f(x::Number, y::Number) = 2x - y


f(2.0, 3.0)


f(2.0, 3)


f(2, 3)

methods(f)


#######################################################################################
##  Method definitions can optionally have type parameters qualifying the signature ##
#######################################################################################

same_type(x::T, y::T) where {T} = true

same_type(x,y) = false

same_type(1, 2)

same_type(1, 2.0)

same_type("foo", "bar")



myappend(v::Vector{T}, x::T) where {T} = [v..., x]

myappend([1,2,3],4)




mytypeof(x::T) where {T} = T

mytypeof(1)

###############################################################################
## One  can put subtype constraints on type parameters in type declarations  ##
###############################################################################

same_type_numeric(x::T, y::T) where {T<:Number} = true

same_type_numeric(x::Number, y::Number) = false


same_type_numeric(1, 2)

same_type_numeric(1, 2.0)

same_type_numeric("foo", "bar")




##  Methods are associated with types, so it is possible to make any arbitrary Julia object "callable"  ##


struct Polynomial{R}
           coeffs::Vector{R}
       end


function (p::Polynomial)(x)
           v = p.coeffs[end]
           for i = (length(p.coeffs)-1):-1:1
               v = v*x + p.coeffs[i]
           end
           return v
       end

p = Polynomial([1,10,100])

p(3)


##################
## COSNTRUCTORS ##
##################

## Constructors are functions that create new objects ##


struct OrderedPair
           x::Real
           y::Real
           OrderedPair(x,y) = x > y ? error("out of order") : new(x,y)
       end


OrderedPair(1, 2)


OrderedPair(2,1)





struct Point{T<:Real}
           x::T
           y::T
       end


Point(1,2)


Point(1.0,2.5)



struct OurRational{T<:Integer} <: Real
           num::T
           den::T
           function OurRational{T}(num::T, den::T) where T<:Integer
               if den == 0
                    error("invalid rational")
               end
               num = flipsign(num, den)
               den = flipsign(den, den)
               g = gcd(num, den)
               num = div(num, g)
               den = div(den, g)
               new(num, den)
           end
       end

OurRational(n::T, d::T) where {T<:Integer} = OurRational{T}(n,d)


OurRational(n::Integer) = OurRational(n,one(n))

import Base.*


*(x::OurRational, y::OurRational) = OurRational(x.num * y.num, x.den * y.den)


a=OurRational(1,2)


b=OurRational(3,4)

a * b


############
## MACROS ##
############


## The : character creates a Symbol

s = :foo

typeof(s)


##  Interpolation is indicated by a prefix $

a = 1;

ex = :($a + b)

$a + b


## A macro maps a tuple of arguments to a returned expression, and the resulting expression is compiled directly rather than requiring a runtime eval call. ##


macro sayhello()
                  return :( println("Hello, world!") )
              end


@sayhello()

function saybye()
       return :( println("Bye, world!") )
       end


saybye()




macro sayhello(name)
           return :( println("Hello, ", $name) )
       end


@sayhello("human")


## Useful examples of macros


 macro assert(ex)
           return :( $ex ? nothing : throw(AssertionError($(string(ex)))) )
       end


 @assert 1 == 1.0


 @assert 1 == 0


macro time(ex)
    return quote
        local t0 = time_ns()
        local val = $ex
        local t1 = time_ns()
        println("elapsed time: ", (t1-t0)/1e9, " seconds")
        val
    end
end

##############################
## ASYNCHRONOUS PROGRAMMING: TASKS ##
##############################




## A Julia Task is a symmetric coroutine

## The task T can be run wheneveer we are ready

t = @task begin; sleep(5); println("done"); end


sleep(10)

istaskstarted(t)


## Add a Task to the scheduler's queue. This causes the task to run constantly when the system is otherwise idle, unless the task performs a blocking operation such as wait. Note that schedule returns immediately

schedule(t);

istaskdone(t)

## Once the task has run, it is no longer runnable

schedule(t);


## The wait function blocks the calling task until some other task finishes.

t = @task begin; sleep(1); println("done"); end
s = @task begin; sleep(1); println("really ddone"); end

schedule(t); schedule(s); wait(s);


## It is common to want to create a task and schedule it right away, so the macro @async is provided for that purpose


t = @async begin;    for i = 1:5  println("done")  end; end;

########################################
## ASYNCHRONOUS PROGRAMMING: CHANNELS  #
########################################

## A Channel is a waitable first-in first-out queue which can have multiple tasks reading from and writing to it.

## This can be used to implement the producer-consumer pattern

function producer(c::Channel)
           put!(c, "start")
           for n=1:4
               put!(c, 2n)
           end
           put!(c, "stop")
       end;


## The Channel constructor accepts a 1-arg function as an argument,
   and wraps this function into a task, bound to the constructed channel.

chnl = Channel(producer);


## One can then take! values repeatedly from the channel object:

take!(chnl)

## Between calls to put!, the producer's execution is suspended and the consumer has control.


take!(chnl)
take!(chnl)
take!(chnl)
take!(chnl);
take!(chnl);

## The channel object is closed automatically when the task terminates.

for x in Channel(producer)
                  println(x)
              end



## A channel can be visualized as a pipe, i.e., it has a write end and a read end.


function producer(c::Channel)
       for i=0:4
           put!(c, i+1)
       end	   
end

c1 = Channel(producer)

function consumer(c::Channel)
       while isready(c1)
           data = take!(c1)
	   result = data^2
	   put!(c,result)
	end
end

c2 = Channel(consumer)


isready(c2)
take!(c2)

isready(c2)
take!(c2)

isready(c2)
take!(c2)

isready(c2)
take!(c2)

isready(c2)
take!(c2)

isready(c2)




## DISTRIBUTED COMPUTING ##


## The first argument to remotecall is the function to call.
The second argument to remotecall is the id of the process that will do the work, and the remaining arguments will be passed to the function being called

r = remotecall(rand, 2, 2, 2)

 s = @spawnat 2 1 .+ fetch(r)

fetch(s)


remotecall_fetch(r-> fetch(r)[1, 1], 2, r)


## To make things easier, the symbol :any can be passed to @spawnat, which picks where to do the operation


r = @spawnat :any rand(2,2)

s = @spawnat :any 1 .+ fetch(r)

fetch(s)


## Code must be available on any process that runs

function rand2(dims...)
           return 2*rand(dims...)
       end

rand2(2,2)

fetch(@spawnat :any rand2(2,2))

@everywhere function rand2(dims...)
           return 2*rand(dims...)
       end

rand2(2,2)

fetch(@spawnat :any rand2(2,2))


## Sending messages and moving data constitute most of the overhead in a distributed program.

A = rand(1000,1000);

Bref = @spawnat :any A^2;

fetch(Bref);


Bref = @spawnat :any rand(1000,1000)^2;

fetch(Bref);



## Fortunately, many useful parallel computations do not require data movement. 

@everywhere function count_heads(n)
           c::Int = 0
           for i = 1:n
               c += rand(Bool)
           end
           c
       end

a = @spawnat :any count_heads(100000000)


b = @spawnat :any count_heads(100000000)

fetch(a)+fetch(b)


The  construction below implements the pattern of assigning iterations to multiple processes, and combining them with a specified reduction (in this case (+)). The result of each iteration is taken as the value of the last expression inside the loop. The whole parallel loop expression itself evaluates to the final answer.

nheads = @distributed (+) for i = 1:200000000
    Int(rand(Bool))
end


Note that although parallel for loops look like serial for loops, their behavior is dramatically different. In particular, the iterations do not happen in a specified order, and writes to variables or arrays will not be globally visible since iterations run on different processes. Any variables used inside the parallel loop will be copied and broadcast to each process.

a = zeros(100000)


@distributed for i = 1:100000
           a[i] = i
       end

a[1]

a[2]

using SharedArrays

a = SharedArray{Float64}(10)
@distributed for i = 1:10
    a[i] = i
end

a[1]

a[2]



## FIBO  ##


@everywhere function fib(n)
                               if (n < 2) 
                                   return n
                               else return fib(n-1) + fib(n-2)
                               end
                            end


z = @spawn fib(10)


fetch(z)


@time [fib(i) for i=1:45];


@everywhere function fib_parallel(n)
                        if (n < 40) 
                            return fib(n)
                        else 
                            x = @spawn fib_parallel(n-1) 
                            y = fib_parallel(n-2)
                            return fetch(x) + y
                        end
                     end


@time [fib_parallel(i) for i=1:42];



@time [fib_parallel(i) for i=1:45];


@time [fib(45) for i=1:4]

@time [fib_parallel(45) for i=1:4]


######################
## Maximum, minimum ##
######################
using Distributed

addprocs(4)


@everywhere function maxnum_serial(a,s,e)
                if s==e
                  a[s]
         else
                   mid = div((s+e),2)
                   low = maxnum_serial(a,s,mid)
                   high = maxnum_serial(a,mid+1,e)
                   low >high ? low : high
                end
       end

@everywhere function maxnum_parallel(a,s,e)
                if (e-s)<=10000000
                  maxnum_serial(a,s,e)
               else
                   mid = div((s+e),2)
                   low_remote = @spawn maxnum_parallel(a,s,mid)
                   high = maxnum_parallel(a,mid+1,e)
                   low = fetch(low_remote)
                   low > high ? low : high
                end
       end

n = 20000000

a=rand(n);

@time maxnum_serial(a,1,n)

## As we can see, the parallel version runs slower than its serial counterpart.


@time maxnum_parallel(a,1,n)

## Indeed, the amount of work (number of comparisons) is in the same order of
magnitude of data transfer (number of integers to move from one processor than another). But the latter costs much more clock-cycles.


@everywhere function minimum_maximum_serial(a,s,e)
                if s==e
                  [a[s], a[s]]
         else
                   mid = div((s+e),2)
                   X = minimum_maximum_serial(a,s,mid)
                   Y = minimum_maximum_serial(a,mid+1,e)
                   [min(X[1],Y[1]), max(X[2],Y[2])]
                end
       end


@everywhere function minimum_maximum_parallel(a,s,e)
                if (e-s)<=10000000
                  minimum_maximum_serial(a,s,e)
               else
                   mid = div((s+e),2)
                   R = @spawn minimum_maximum_parallel(a,s,mid)
                   Y = minimum_maximum_parallel(a,mid+1,e)
                   X = fetch(R)
                   [min(X[1],Y[1]), max(X[2],Y[2])]
                end
       end
       
n = 20000000
a=rand(n);

@time minimum_maximum_serial(a,1,n)

@time minimum_maximum_parallel(a,1,n)

## Should be satidified with the above experimental results

## No, since computing serially the minimum alone is about 10 times faster


function mergesort(data, istart, iend)
                      if(istart < iend)
                              mid = (istart + iend) >>>1
                              mergesort(data, istart, mid)
                              mergesort(data, mid+1, iend)
                                 merge(data, istart, mid, iend)
                      end
              end

function merge( data, istart, mid, iend)
                      n = iend - istart + 1
                      temp = zeros(n)
                      s = istart
                      m = mid+1
                      for tem = 1:n
                              if s <= mid && (m > iend || data[s] <= data[m])
                                      temp[tem] = data[s]
                                      s=s+1
                              else
                                      temp[tem] = data[m]
                                      m=m+1
                              end
                      end
                      data[istart:iend] = temp[1:n]
              end

n = 1000000
A = [rem(rand(Int32),10) for i =1:n];
@time mergesort(A, 1, n);

## Towards paralle code and pretending that we do not know about SharedArrays, we write an uut-of-place serial merge sort

function mergesort(data, istart, iend)
                      if(istart < iend)
                              mid = div((istart + iend),2)
                              a = mergesort(data, istart, mid)
                              b = mergesort(data,mid+1, iend)
                              c = merge(a, b, istart, mid, iend)
                      else
                          [data[istart]]
                      end
              end

@everywhere function merge(a, b, istart, mid, iend)
                      n = iend - istart + 1
                      nb = iend - mid
                      na = mid - istart + 1
                      c = zeros(n)
                      s = 1
                      m = 1
                      for tem = 1:n
                              if s <= na && (m > nb || a[s] <= b[m])
                                      c[tem] = a[s]
                                      s= s+1
                              else
                                      c[tem] = b[m]
                                      m=m+1
                              end
                      end
                      c
              end

n = 1000000;
A = [rem(rand(Int32),10) for i =1:n];
@time mergesort(A, 1, n);


@everywhere function mergesort_serial(data, istart, iend)
                      if(istart < iend)
                              mid = div((istart + iend),2)
                              a = mergesort_serial(data, istart, mid)
                              b = mergesort_serial(data,mid+1, iend)
                              c = merge(a, b, istart, mid, iend)
                      else
                          [data[istart]]
                      end
              end

@everywhere function mergesort_parallel(data, istart, iend)
                      if(iend - istart <= 2500000)
                              mergesort_serial(data, istart, iend)
                       else
                              mid = div((istart + iend),2)
                              a = @spawn mergesort_parallel(data, istart, mid)
                              b = mergesort_parallel(data,mid+1, iend)
                              c = merge(fetch(a), b, istart, mid, iend)
                      end
              end
	      
n = 10000000;
A = [rem(rand(Int32),10) for i =1:n];
@time mergesort_serial(A, 1, n);
@time mergesort_parallel(A, 1, n);

## So we got some speedup!

#########
## MMM ##
#########

using Distributed

addprocs(4)


function dacmm(i0, i1, j0, j1, k0, k1, A, B, C, n, basecase)
           ## A, B, C are matrices
           ## We compute C = A * B

           if n > basecase
              n = div(n, 2)
              dacmm(i0, i1, j0, j1, k0, k1, A, B, C, n, basecase)
              dacmm(i0, i1, j0, j1+n, k0, k1+n, A, B, C, n, basecase)
              dacmm(i0+n, i1, j0, j1, k0+n, k1, A, B, C, n, basecase)
              dacmm(i0+n, i1, j0, j1+n, k0+n, k1+n, A, B, C, n, basecase)
              dacmm(i0, i1+n, j0+n, j1, k0, k1, A, B, C, n, basecase)
              dacmm(i0, i1+n, j0+n, j1+n, k0, k1+n, A, B, C, n, basecase)
              dacmm(i0+n, i1+n, j0+n, j1, k0+n, k1, A, B, C, n, basecase)
              dacmm(i0+n, i1+n, j0+n, j1+n, k0+n, k1+n, A, B, C, n, basecase)
           else
             for i= 1:n, j=1:n, k=1:n
                 C[i+k0,k1+j] = C[i+k0,k1+j] + A[i+i0,i1+k] * B[k+j0,j1+j]
             end
           end
       end




n=4
basecase = 2
A = [rem(rand(Int32),5) for i =1:n, j = 1:n]
B = [rem(rand(Int32),5) for i =1:n, j = 1:n]
C = zeros(Int32,n,n);
dacmm(0, 0, 0, 0, 0, 0, A, B, C, n, basecase)
A * B - C


function test_dacmm(n, basecase)
	 A = [rem(rand(Int32),5) for i =1:n, j = 1:n]
	 B = [rem(rand(Int32),5) for i =1:n, j = 1:n]
	 C = zeros(Int32,n,n);
	 @time dacmm(0, 0, 0, 0, 0, 0, A, B, C, n, basecase)
	 if n < 1024
	 return A * B - C
	 else
	 @assert A* B == C
	 end
end


test_dacmm(4, 2)

test_dacmm(16, 4)

test_dacmm(256, 16)

test_dacmm(1024, 2)

[test_dacmm(1024,2^i) for i=1:10]





@everywhere function dacmm_parallel(i0, i1, j0, j1, k0, k1, A, B, C, n, basecase)
   l = []
   if n > basecase && nprocs() > 3
    n = div(n,2)
    lrf = [
       @spawnat procs()[1] dacmm_parallel(i0, i1, j0, j1, k0, k1, A, B, C, n,basecase),
       @spawnat procs()[2] dacmm_parallel(i0, i1, j0, j1+n, k0, k1+n, A, B, C, n,basecase),
       @spawnat procs()[3] dacmm_parallel(i0+n, i1, j0, j1, k0+n, k1, A, B, C, n,basecase),
       @spawnat procs()[4] dacmm_parallel(i0+n, i1, j0, j1+n, k0+n, k1+n, A, B, C, n, basecase)
       ]
    ## pmap(fetch, lrf)
    l = [l..., map(fetch, lrf)]
    lrf = [
       @spawnat procs()[1] dacmm_parallel(i0, i1+n, j0+n, j1, k0, k1, A, B, C, n,basecase),
       @spawnat procs()[2] dacmm_parallel(i0, i1+n, j0+n, j1+n, k0, k1+n, A, B, C, n, basecase),
       @spawnat procs()[3] dacmm_parallel(i0+n, i1+n, j0+n, j1, k0+n, k1, A, B, C, n, basecase),
       @spawnat procs()[4] dacmm_parallel(i0+n, i1+n, j0+n, j1+n, k0+n, k1+n, A, B, C, n, basecase)
       ]
    ## pmap(fetch, lrf)
    l = [l..., map(fetch, lrf)]
   else
    for i= 1:n, j=1:n, k=1:n
        C[i+k0,k1+j] += A[i+i0,i1+k] * B[k+j0,j1+j]
    end
   end
   return l
end


using SharedArrays

@everywhere n = 4
a = SharedArray{Int32}(n,n)
for i = 1:n, j = 1:n a[i,j]=rem(rand(Int64),5) end
b = SharedArray{Int32}(n,n)
for i = 1:n, j = 1:n b[i,j]=rem(rand(Int64),5) end
c = SharedArray{Int32}(n,n)
dacmm_parallel(0,0,0,0,0,0,a,b,c,n,n)
c - a * b

c = SharedArray{Int32}(n,n)
dacmm_parallel(0,0,0,0,0,0,a,b,c,n,div(n,2))
c - a * b

function test_dacmm_parallel(n, basecase)
	 a = SharedArray{Int32}(n,n)
	 for i = 1:n, j = 1:n a[i,j]=rem(rand(Int32),5) end
	 b = SharedArray{Int32}(n,n)
	 for i = 1:n, j = 1:n b[i,j]=rem(rand(Int32),5) end
	 c = SharedArray{Int32}(n,n)
	 @time dacmm_parallel(0,0,0,0,0,0,a,b,c,n,basecase)
	 if n < 128
	    sleep(2)
	    return a * b - c
	 end
end

@everywhere n = 16
test_dacmm_parallel(n,n)
test_dacmm_parallel(n,div(n,2))
test_dacmm_parallel(n,div(n,4))


@everywhere n = 64
test_dacmm_parallel(n,n)
test_dacmm_parallel(n,div(n,2))
test_dacmm_parallel(n,div(n,4))
test_dacmm_parallel(n,div(n,8))


@everywhere n = 128
[test_dacmm_parallel(n,2^i) for i=2:6]

@everywhere n = 256
[test_dacmm_parallel(n,2^i) for i=3:7]

@everywhere n = 1024
[test_dacmm_parallel(n,2^i) for i=4:10]

[test_dacmm(n,2^i) for i=3:10]