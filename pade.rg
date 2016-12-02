import "regent"

local c     = regentlib.c
local cmath = terralib.includec("math.h")
local PI    = cmath.M_PI

local max = regentlib.fmax

-- Some problem parameters
local NN = 256
local LL = 2.0*math.pi
local DX = LL / NN
local DY = LL / NN
local DZ = LL / NN
local ONEBYDX = 1.0 / (DX)
local ONEBYDY = 1.0 / (DY)
local ONEBYDZ = 1.0 / (DZ)
local a10d1 = (17.0/12.0)/2.0
local b10d1 = (101.0/150.0)/4.0
local c10d1 = (1.0/100.0)/6.0

fspace coordinates {
  x   : double,
  y   : double,
  z   : double,
}

fspace point {
  f   : double,
  dfx : double,
  dfy : double,
  dfz : double,
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
  reads writes(points.dfx), reads(LU)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.x + 1

  for j = bounds.lo.y, bounds.hi.y+1 do
    for k = bounds.lo.z, bounds.hi.z+1 do

      -- Step 8
      points[{1,j,k}].dfx = points[{1,j,k}].dfx - LU[1].b*points[{0,j,k}].dfx
      var sum1 : double = LU[0].k*points[{0,j,k}].dfx + LU[1].k*points[{1,j,k}].dfx
      var sum2 : double = LU[0].l*points[{0,j,k}].dfx + LU[1].l*points[{1,j,k}].dfx

      -- Step 9
      for i = 2,N-2 do
        points[{i,j,k}].dfx = points[{i,j,k}].dfx - LU[i].b*points[{i-1,j,k}].dfx - LU[i].eg*points[{i-2,j,k}].dfx
        sum1 += LU[i].k*points[{i,j,k}].dfx
        sum2 += LU[i].l*points[{i,j,k}].dfx
      end

      -- Step 10
      points[{N-2,j,k}].dfx = points[{N-2,j,k}].dfx - sum1
      points[{N-1,j,k}].dfx = ( points[{N-1,j,k}].dfx - sum2 - LU[N-2].l*points[{N-2,j,k}].dfx )*LU[N-1].g

      -- Step 11
      points[{N-2,j,k}].dfx = ( points[{N-2,j,k}].dfx - LU[N-2].w*points[{N-1,j,k}].dfx )*LU[N-2].g
      points[{N-3,j,k}].dfx = ( points[{N-3,j,k}].dfx - LU[N-3].v*points[{N-2,j,k}].dfx - LU[N-3].w*points[{N-1,j,k}].dfx )*LU[N-3].g
      points[{N-4,j,k}].dfx = ( points[{N-4,j,k}].dfx - LU[N-4].h*points[{N-3,j,k}].dfx - LU[N-4].v*points[{N-2,j,k}].dfx - LU[N-4].w*points[{N-1,j,k}].dfx )*LU[N-4].g
      for i = N-5,-1,-1 do
        points[{i,j,k}].dfx = ( points[{i,j,k}].dfx - LU[i].h*points[{i+1,j,k}].dfx - LU[i].ff*points[{i+2,j,k}].dfx - LU[i].v*points[{N-2,j,k}].dfx - LU[i].w*points[{N-1,j,k}].dfx )*LU[i].g
      end

    end
  end
  return 1
end

task SolveYLU( points : region(ispace(int3d), point),
               LU     : region(ispace(int1d), LU_struct) )
where
  reads writes(points.dfy), reads(LU)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.y + 1

  for i = bounds.lo.x, bounds.hi.x+1 do
    for k = bounds.lo.z, bounds.hi.z+1 do

      -- Step 8
      points[{i,1,k}].dfy = points[{i,1,k}].dfy - LU[1].b*points[{i,0,k}].dfy
      var sum1 : double = LU[0].k*points[{i,0,k}].dfy + LU[1].k*points[{i,1,k}].dfy
      var sum2 : double = LU[0].l*points[{i,0,k}].dfy + LU[1].l*points[{i,1,k}].dfy

      -- Step 9
      for j = 2,N-2 do
        points[{i,j,k}].dfy = points[{i,j,k}].dfy - LU[j].b*points[{i,j-1,k}].dfy - LU[j].eg*points[{i,j-2,k}].dfy
        sum1 += LU[j].k*points[{i,j,k}].dfy
        sum2 += LU[j].l*points[{i,j,k}].dfy
      end

      -- Step 10
      points[{i,N-2,k}].dfy = points[{i,N-2,k}].dfy - sum1
      points[{i,N-1,k}].dfy = ( points[{i,N-1,k}].dfy - sum2 - LU[N-2].l*points[{i,N-2,k}].dfy )*LU[N-1].g

      -- Step 11
      points[{i,N-2,k}].dfy = ( points[{i,N-2,k}].dfy - LU[N-2].w*points[{i,N-1,k}].dfy )*LU[N-2].g
      points[{i,N-3,k}].dfy = ( points[{i,N-3,k}].dfy - LU[N-3].v*points[{i,N-2,k}].dfy - LU[N-3].w*points[{i,N-1,k}].dfy )*LU[N-3].g
      points[{i,N-4,k}].dfy = ( points[{i,N-4,k}].dfy - LU[N-4].h*points[{i,N-3,k}].dfy - LU[N-4].v*points[{i,N-2,k}].dfy - LU[N-4].w*points[{i,N-1,k}].dfy )*LU[N-4].g
      for j = N-5,-1,-1 do
        points[{i,j,k}].dfy = ( points[{i,j,k}].dfy - LU[j].h*points[{i,j+1,k}].dfy - LU[j].ff*points[{i,j+2,k}].dfy - LU[j].v*points[{i,N-2,k}].dfy - LU[j].w*points[{i,N-1,k}].dfy )*LU[j].g
      end

    end
  end
  return 1
end

task SolveZLU( points : region(ispace(int3d), point),
               LU     : region(ispace(int1d), LU_struct) )
where
  reads writes(points.dfz), reads(LU)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.z + 1

  for i = bounds.lo.x, bounds.hi.x+1 do
    for j = bounds.lo.y, bounds.hi.y+1 do

      -- Step 8
      points[{i,j,1}].dfz = points[{i,j,1}].dfz - LU[1].b*points[{i,j,0}].dfz
      var sum1 : double = LU[0].k*points[{i,j,0}].dfz + LU[1].k*points[{i,j,1}].dfz
      var sum2 : double = LU[0].l*points[{i,j,0}].dfz + LU[1].l*points[{i,j,1}].dfz

      -- Step 9
      for k = 2,N-2 do
        points[{i,j,k}].dfz = points[{i,j,k}].dfz - LU[k].b*points[{i,j,k-1}].dfz - LU[k].eg*points[{i,j,k-2}].dfz
        sum1 += LU[k].k*points[{i,j,k}].dfz
        sum2 += LU[k].l*points[{i,j,k}].dfz
      end

      -- Step 10
      points[{i,j,N-2}].dfz = points[{i,j,N-2}].dfz - sum1
      points[{i,j,N-1}].dfz = ( points[{i,j,N-1}].dfz - sum2 - LU[N-2].l*points[{i,j,N-2}].dfz )*LU[N-1].g

      -- Step 11
      points[{i,j,N-2}].dfz = ( points[{i,j,N-2}].dfz - LU[N-2].w*points[{i,j,N-1}].dfz )*LU[N-2].g
      points[{i,j,N-3}].dfz = ( points[{i,j,N-3}].dfz - LU[N-3].v*points[{i,j,N-2}].dfz - LU[N-3].w*points[{i,j,N-1}].dfz )*LU[N-3].g
      points[{i,j,N-4}].dfz = ( points[{i,j,N-4}].dfz - LU[N-4].h*points[{i,j,N-3}].dfz - LU[N-4].v*points[{i,j,N-2}].dfz - LU[N-4].w*points[{i,j,N-1}].dfz )*LU[N-4].g
      for k = N-5,-1,-1 do
        points[{i,j,k}].dfz = ( points[{i,j,k}].dfz - LU[k].h*points[{i,j,k+1}].dfz - LU[k].ff*points[{i,j,k+2}].dfz - LU[k].v*points[{i,j,N-2}].dfz - LU[k].w*points[{i,j,N-1}].dfz )*LU[k].g
      end

    end
  end
  return 1
end

local function poff(i, x, y, z, N)
  return rexpr int3d { x = (i.x + x + N)%N, y = (i.y + y + N)%N, z = (i.z + z + N)%N } end
end

local function make_stencil_pattern(points, index, a10, b10, c10, N, onebydx, dir)
  local value

  if dir == 0 then      -- x direction stencil
    value = rexpr       - c10*points[ [poff(index, -3, 0, 0, N)] ].f end
    value = rexpr value - b10*points[ [poff(index, -2, 0, 0, N)] ].f end
    value = rexpr value - a10*points[ [poff(index, -1, 0, 0, N)] ].f end
    value = rexpr value + a10*points[ [poff(index,  1, 0, 0, N)] ].f end
    value = rexpr value + b10*points[ [poff(index,  2, 0, 0, N)] ].f end
    value = rexpr value + c10*points[ [poff(index,  3, 0, 0, N)] ].f end
    value = rexpr onebydx * ( value ) end
  elseif dir == 1 then  -- y direction stencil
    value = rexpr       - c10*points[ [poff(index, 0, -3, 0, N)] ].f end
    value = rexpr value - b10*points[ [poff(index, 0, -2, 0, N)] ].f end
    value = rexpr value - a10*points[ [poff(index, 0, -1, 0, N)] ].f end
    value = rexpr value + a10*points[ [poff(index, 0,  1, 0, N)] ].f end
    value = rexpr value + b10*points[ [poff(index, 0,  2, 0, N)] ].f end
    value = rexpr value + c10*points[ [poff(index, 0,  3, 0, N)] ].f end
    value = rexpr onebydx * ( value ) end
  elseif dir == 2 then  -- z direction stencil
    value = rexpr       - c10*points[ [poff(index, 0, 0, -3, N)] ].f end
    value = rexpr value - b10*points[ [poff(index, 0, 0, -2, N)] ].f end
    value = rexpr value - a10*points[ [poff(index, 0, 0, -1, N)] ].f end
    value = rexpr value + a10*points[ [poff(index, 0, 0,  1, N)] ].f end
    value = rexpr value + b10*points[ [poff(index, 0, 0,  2, N)] ].f end
    value = rexpr value + c10*points[ [poff(index, 0, 0,  3, N)] ].f end
    value = rexpr onebydx * ( value ) end
  end
  return value
end

local function make_stencil_x(N, onebydx, a10, b10, c10)
  local task rhs_x( points : region(ispace(int3d), point) )
  where
    reads(points.f), writes(points.dfx)
  do
    for i in points do
      points[i].dfx = [make_stencil_pattern(points, i, a10, b10, c10, N, onebydx, 0)]
    end
  end
  return rhs_x
end

local function make_stencil_y(N, onebydy, a10, b10, c10)
  local task rhs_y( points : region(ispace(int3d), point) )
  where
    reads(points.f), writes(points.dfy)
  do
    for i in points do
      points[i].dfy = [make_stencil_pattern(points, i, a10, b10, c10, N, onebydy, 1)]
    end
  end
  return rhs_y
end

local function make_stencil_z(N, onebydz, a10, b10, c10, dir)
  local task rhs_z( points : region(ispace(int3d), point) )
  where
    reads(points.f), writes(points.dfz)
  do
    for i in points do
      points[i].dfz = [make_stencil_pattern(points, i, a10, b10, c10, N, onebydz, 2)]
    end
  end
  return rhs_z
end

local ComputeXRHS = make_stencil_x(NN, ONEBYDX, a10d1, b10d1, c10d1)
local ComputeYRHS = make_stencil_y(NN, ONEBYDY, a10d1, b10d1, c10d1)
local ComputeZRHS = make_stencil_z(NN, ONEBYDZ, a10d1, b10d1, c10d1)

task ddx( points : region(ispace(int3d), point),
          LU     : region(ispace(int1d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfx)
do
  ComputeXRHS(points)
  var token = SolveXLU(points,LU)
  return token
end

task ddy( points : region(ispace(int3d), point),
          LU     : region(ispace(int1d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfy)
do
  ComputeYRHS(points)
  var token = SolveYLU(points,LU)
  return token
end

task ddz( points : region(ispace(int3d), point),
          LU     : region(ispace(int1d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfz)
do
  ComputeZRHS(points)
  var token = SolveZLU(points,LU)
  return token
end

task initialize( points : region(ispace(int3d), point),
                 exact  : region(ispace(int3d), point),
                 coords : region(ispace(int3d), coordinates),
                 dx     : double,
                 dy     : double,
                 dz     : double )
where
  reads writes(coords.x, coords.y, coords.z, points.f, exact.dfx, exact.dfy, exact.dfz)
do
  var bounds = points.ispace.bounds

  for i = bounds.lo.x, bounds.hi.x+1 do
    for j = bounds.lo.y, bounds.hi.y+1 do
      for k = bounds.lo.z, bounds.hi.z+1 do
        var e : int3d = { x = i, y = j, z = k }
        coords[e].x   = i*dx
        coords[e].y   = j*dy
        coords[e].z   = k*dz
        points[e].f   = cmath.sin(coords[e].x) + cmath.sin(coords[e].y) + cmath.sin(coords[e].z)
        exact [e].dfx = cmath.cos(coords[e].x)
        exact [e].dfy = cmath.cos(coords[e].y)
        exact [e].dfz = cmath.cos(coords[e].z)
      end
    end
  end
  return 0
end

task get_error_x( points : region(ispace(int3d), point),
                  exact  : region(ispace(int3d), point) )
where
  reads(points.dfx, exact.dfx)
do
  var err : double = 0.0
  for i in points do
    err = max(err, cmath.fabs(points[i].dfx - exact[i].dfx))
  end
  return err
end

task get_error_y( points : region(ispace(int3d), point),
                  exact  : region(ispace(int3d), point) )
where
  reads(points.dfy, exact.dfy)
do
  var err : double = 0.0
  for i in points do
    err = max(err, cmath.fabs(points[i].dfy - exact[i].dfy))
  end
  return err
end

task get_error_z( points : region(ispace(int3d), point),
                  exact  : region(ispace(int3d), point) )
where
  reads(points.dfz, exact.dfz)
do
  var err : double = 0.0
  for i in points do
    err = max(err, cmath.fabs(points[i].dfz - exact[i].dfz))
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
  var dy : double = DY      -- Grid spacing
  var dz : double = DZ      -- Grid spacing

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
  var coords = region(grid, coordinates)
  var points = region(grid, point)
  var exact  = region(grid, point)
  
  -- Initialize function f
  var token = initialize(points, exact, coords, dx, dy, dz)

  wait_for(token)
  var ts_start = c.legion_get_current_time_in_micros()
  
  -- Get df/dx
  -- ComputeXRHS(points)
  -- token += SolveXLU(points,LU_x)
  token += ddx(points,LU_x)
  
  -- wait_for(token)
  -- var ts_x = c.legion_get_current_time_in_micros() - ts_start
  -- wait_for(ts_x)
  -- ts_start = c.legion_get_current_time_in_micros()
  
  -- ComputeYRHS(points)
  -- token += SolveYLU(points,LU_x)
  token += ddy(points,LU_x)
  
  -- wait_for(token)
  -- var ts_y = c.legion_get_current_time_in_micros() - ts_start
  -- wait_for(ts_y)
  -- ts_start = c.legion_get_current_time_in_micros()
  
  -- ComputeZRHS(points)
  -- token += SolveZLU(points,LU_x)
  token += ddz(points,LU_x)
  
  wait_for(token)
  var ts_z = c.legion_get_current_time_in_micros() - ts_start
  
  var err_x = get_error_x(points,exact) 
  var err_y = get_error_y(points,exact) 
  var err_z = get_error_z(points,exact) 

  -- c.printf("Time to get the x derivative: %12.5e\n", (ts_x)*1e-6)
  c.printf("Maximum error in x = %12.5e\n", err_x)
  -- c.printf("Time to get the y derivative: %12.5e\n", (ts_y)*1e-6)
  c.printf("Maximum error in y = %12.5e\n", err_y)
  c.printf("Time to get the z derivative: %12.5e\n", (ts_z)*1e-6)
  c.printf("Maximum error in z = %12.5e\n", err_z)

end

regentlib.start(main)
