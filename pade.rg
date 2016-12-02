import "regent"

local c     = regentlib.c
local cmath = terralib.includec("math.h")
local PI    = cmath.M_PI

local max = regentlib.fmax

-- Some problem parameters
local NN = 256
local LL = 2.0*math.pi
local DX = LL / NN
local ONEBYDX = 1.0 / (DX)
local a10d1 = (17.0/12.0)/2.0
local b10d1 = (101.0/150.0)/4.0
local c10d1 = (1.0/100.0)/6.0

fspace point {
  x   : double,
  f   : double,
  df  : double,
  dfe : double,
}

fspace LU_struct {
  b  : double,
  eg : double,
  k  : double,
  l  : double,
  g  : double,
  h  : double,
  ff : double,
  v  : double,
  w  : double,
}

task get_LU_decomposition(LU : region(ispace(int1d), LU_struct),
                          e  : double,
                          a  : double,
                          d  : double,
                          cc : double,
                          f  : double)
where
  reads writes( LU )
do

  var N : int64 = LU.ispace.bounds.hi + 1

  -- Step 1
  LU[0].g = d
  LU[1].b = a/LU[0].g
  LU[0].h = cc
  LU[0].k = f/LU[0].g
  LU[0].w = a
  LU[0].v = e
  LU[0].l = cc/LU[0].g
  
  LU[1].g = d - LU[1].b*LU[0].h
  LU[1].k = -LU[0].k*LU[0].h/LU[1].g
  LU[1].w = e - LU[1].b*LU[0].w
  LU[1].v = -LU[1].b*LU[0].v
  LU[1].l = (f - LU[0].l*LU[0].h) / LU[1].g
  LU[1].h = cc - LU[1].b*f

  -- Step 2
  for i = 2,N-3 do
    LU[{i}].b = ( a - ( e/LU[{i-2}].g )*LU[{i-2}].h ) / LU[{i-1}].g
    LU[{i}].h = cc - LU[{i}].b*f
    LU[{i}].g = d - ( e/LU[{i-2}].g )*f - LU[{i}].b*LU[{i-1}].h
  end

  -- Step 3
  LU[N-3].b = ( a - ( e/LU[N-5].g )*LU[N-5].h ) / LU[N-4].g
  LU[N-3].g = d - ( e/LU[N-5].g )*f - LU[N-3].b*LU[N-4].h

  -- Step 4
  for i = 2,N-4 do
    LU[i].k = -( LU[i-2].k*f + LU[i-1].k*LU[i-1].h ) / LU[i].g
    LU[i].v = -( e/LU[i-2].g )*LU[i-2].v - LU[i].b*LU[i-1].v
  end

  -- Step 5
  LU[N-4].k = ( e - LU[N-6].k*f - LU[N-5].k*LU[N-5].h ) / LU[N-4].g
  LU[N-3].k = ( a - LU[N-5].k*f - LU[N-4].k*LU[N-4].h ) / LU[N-3].g
  LU[N-4].v = f  - ( e/LU[N-6].g )*LU[N-6].v - LU[N-4].b*LU[N-5].v
  LU[N-3].v = cc - ( e/LU[N-5].g )*LU[N-5].v - LU[N-3].b*LU[N-4].v
  LU[N-2].g = d
  for i = 0,N-2 do
    LU[N-2].g -= LU[i].k*LU[i].v
  end

  -- Step 6
  for i = 2,N-3 do
    LU[i].w = -( e/LU[i-2].g )*LU[i-2].w - LU[i].b*LU[i-1].w
    LU[i].l = -( LU[i-2].l*f + LU[i-1].l*LU[i-1].h ) / LU[i].g
  end

  -- Step 7
  LU[N-3].w = f - ( e/LU[N-5].g )*LU[N-5].w - LU[N-3].b*LU[N-4].w
  LU[N-2].w = cc
  for i = 0,N-2 do
    LU[N-2].w -= LU[i].k*LU[i].w
  end
  LU[N-3].l = ( e - LU[N-5].l*f - LU[N-4].l*LU[N-4].h ) / LU[N-3].g
  LU[N-2].l = a
  for i = 0,N-2 do
    LU[N-2].l -= LU[i].l*LU[i].v
  end
  LU[N-2].l = LU[N-2].l / LU[N-2].g
  LU[N-1].g = d
  for i = 0,N-1 do
    LU[N-1].g -= LU[i].l*LU[i].w
  end

  -- Set eg = e/g
  for i = 2,N-2 do
    LU[i].eg = e/LU[i-2].g
  end

  -- Set ff = f
  for i = 0,N-4 do
    LU[i].ff = f
  end

  -- Set g = 1/g
  for i = 0,N do
    LU[i].g = 1.0/LU[i].g
  end

  -- c.printf("LU decomposition:\n")
  -- for i = 0,N do
  --   c.printf("%8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f\n",LU[i].b,LU[i].eg,LU[i].k,LU[i].l,LU[i].g,LU[i].h,LU[i].ff,LU[i].v,LU[i].w)
  -- end

end

task SolveXLU( points : region(ispace(int3d), point),
               LU     : region(ispace(int1d), LU_struct) )
where
  reads writes(points.df), reads(LU)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.x + 1

  for j = bounds.lo.y, bounds.hi.y+1 do
    for k = bounds.lo.z, bounds.hi.z+1 do

      -- Step 8
      points[{1,j,k}].df = points[{1,j,k}].df - LU[1].b*points[{0,j,k}].df
      var sum1 : double = LU[0].k*points[{0,j,k}].df + LU[1].k*points[{1,j,k}].df
      var sum2 : double = LU[0].l*points[{0,j,k}].df + LU[1].l*points[{1,j,k}].df

      -- Step 9
      for i = 2,N-2 do
        points[{i,j,k}].df = points[{i,j,k}].df - LU[i].b*points[{i-1,j,k}].df - LU[i].eg*points[{i-2,j,k}].df
        sum1 += LU[i].k*points[{i,j,k}].df
        sum2 += LU[i].l*points[{i,j,k}].df
      end

      -- Step 10
      points[{N-2,j,k}].df = points[{N-2,j,k}].df - sum1
      points[{N-1,j,k}].df = ( points[{N-1,j,k}].df - sum2 - LU[N-2].l*points[{N-2,j,k}].df )*LU[N-1].g

      -- Step 11
      points[{N-2,j,k}].df = ( points[{N-2,j,k}].df - LU[N-2].w*points[{N-1,j,k}].df )*LU[N-2].g
      points[{N-3,j,k}].df = ( points[{N-3,j,k}].df - LU[N-3].v*points[{N-2,j,k}].df - LU[N-3].w*points[{N-1,j,k}].df )*LU[N-3].g
      points[{N-4,j,k}].df = ( points[{N-4,j,k}].df - LU[N-4].h*points[{N-3,j,k}].df - LU[N-4].v*points[{N-2,j,k}].df - LU[N-4].w*points[{N-1,j,k}].df )*LU[N-4].g
      for i = N-5,-1,-1 do
        points[{i,j,k}].df = ( points[{i,j,k}].df - LU[i].h*points[{i+1,j,k}].df - LU[i].ff*points[{i+2,j,k}].df - LU[i].v*points[{N-2,j,k}].df - LU[i].w*points[{N-1,j,k}].df )*LU[i].g
      end

    end
  end
  return 1
end

local function poff(i, x, y, z, N)
  return rexpr int3d { x = (i.x + x + N)%N, y = (i.y + y + N)%N, z = (i.z + z + N)%N } end
end

local function make_stencil_pattern(points, index, a10, b10, c10, N, onebydx)
  local value
  value = rexpr       - c10*points[ [poff(index, -3, 0, 0, N)] ].f end
  value = rexpr value - b10*points[ [poff(index, -2, 0, 0, N)] ].f end
  value = rexpr value - a10*points[ [poff(index, -1, 0, 0, N)] ].f end
  value = rexpr value + a10*points[ [poff(index,  1, 0, 0, N)] ].f end
  value = rexpr value + b10*points[ [poff(index,  2, 0, 0, N)] ].f end
  value = rexpr value + c10*points[ [poff(index,  3, 0, 0, N)] ].f end
  value = rexpr onebydx * ( value ) end
  return value
end

local function make_stencil(N, onebydx, a10, b10, c10)
  local task rhs( points : region(ispace(int3d), point) )
  where
    reads(points.f), writes(points.df)
  do
    for i in points do
      points[i].df = [make_stencil_pattern(points, i, a10, b10, c10, N, onebydx)]
    end
  end
  return rhs
end

local ComputeXRHS = make_stencil(NN, ONEBYDX, a10d1, b10d1, c10d1)

task initialize( points : region(ispace(int3d), point),
                 dx     : double )
where
  reads writes(points.x, points.f, points.dfe)
do
  var bounds = points.ispace.bounds

  for i = bounds.lo.x, bounds.hi.x+1 do
    for j = bounds.lo.y, bounds.hi.y+1 do
      for k = bounds.lo.z, bounds.hi.z+1 do
        var e : int3d = { x = i, y = j, z = k }
        points[e].x   = i*dx
        points[e].f   = cmath.sin(points[e].x)
        points[e].dfe = cmath.cos(points[e].x)
      end
    end
  end
  return 0
end

task ddx( points : region(ispace(int3d), point),
          dx     : double )
where
  reads(points.f), writes(points.df)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.x + 1
  
  for i = bounds.lo.x, bounds.hi.x+1 do
    for j = bounds.lo.y, bounds.hi.y+1 do
      for k = bounds.lo.z, bounds.hi.z+1 do
        var e : int3d = { x = i, y = j, z = k }
        points[ { x = i, y = j, z = k } ].df = ( points[ { x = (i+1+N)%N, y = j, z = k } ].f - points[ { x = (i-1+N)%N, y = j, z = k } ].f ) / (2.0*dx)
      end
    end
  end

end

task get_error( points : region(ispace(int3d), point) )
where
  reads(points.df, points.dfe)
do
  var err : double = 0.0
  for i in points do
    err = max(err, cmath.fabs(points[i].df - points[i].dfe))
  end
  return err
end

terra wait_for(x : int)
  return x
end

task main()
  var L  : double = LL      -- Domain length
  var N  : int64  = NN      -- Grid size
  var dx : double = DX      -- Grid spacing

  c.printf("================ Problem parameters ================\n")
  c.printf("                   N  = %d\n", N )
  c.printf("                   L  = %f\n", L )
  c.printf("                   dx = %f\n", dx)
  c.printf("====================================================\n")

  -- Coefficients for the 10th order 1st derivative
  var alpha10d1 : double = 1.0/2.0
  var beta10d1  : double = 1.0/20.0

  var grid_x = ispace(int1d, N)
  var LU_x   = region(grid_x, LU_struct)

  get_LU_decomposition(LU_x, beta10d1, alpha10d1, 1.0, alpha10d1, beta10d1)

  var grid   = ispace(int3d, { x = N, y = N, z = N })
  var points = region(grid, point)
  
  -- Initialize function f
  var token = initialize(points, dx)

  wait_for(token)
  var ts_start = c.legion_get_current_time_in_micros()
  
  -- Get df/dx
  ComputeXRHS(points)
  token = SolveXLU(points,LU_x)
  
  wait_for(token)
  var ts_end = c.legion_get_current_time_in_micros()

  c.printf("Time to get the x derivative: %12.5e\n", (ts_end-ts_start)*1e-6)
  c.printf("Maximum error = %12.5e\n", get_error(points))

  -- for i = 0, N do
  --   var e : int3d = { x = i, y = 0, z = 0 }
  --   c.printf("i = %3d, x = %8.5f, f = %8.5f, df = %8.5f, error = %12.5e\n", 
  --             i, points[e].x, points[e].f, points[e].df, (points[e].df-points[e].dfe))
  -- end

end

regentlib.start(main)
