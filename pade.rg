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

local parallelism = 4

local a10d1 = ( 17.0/ 12.0)/2.0
local b10d1 = (101.0/150.0)/4.0
local c10d1 = (  1.0/100.0)/6.0

local a10d2 = (1065.0/1798.0)/1.0
local b10d2 = (1038.0/ 899.0)/4.0
local c10d2 = (  79.0/1798.0)/9.0

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

task factorize(parallelism : int) : int2d
  var limit = [int](cmath.sqrt([double](parallelism)))
  var size_x = 1
  var size_y = parallelism
  for i = 1, limit + 1 do
    if parallelism % i == 0 then
      size_x, size_y = i, parallelism / i
      if size_x > size_y then
        size_x, size_y = size_y, size_x
      end
    end
  end
  return int2d { size_x, size_y }
end

task make_xpencil( points  : region(ispace(int3d), point),
                   xpencil : ispace(int2d) )
  var coloring = c.legion_domain_point_coloring_create()

  var prow = xpencil.bounds.hi.x + 1
  var pcol = xpencil.bounds.hi.y + 1

  var bounds = points.ispace.bounds
  var Nx = bounds.hi.x + 1
  var Ny = bounds.hi.y + 1
  var Nz = bounds.hi.z + 1

  for i in xpencil do
    var lo = int3d { x = 0, y = i.x*(Ny/prow), z = i.y*(Nz/pcol) }
    var hi = int3d { x = Nx-1, y = (i.x+1)*(Ny/prow)-1, z = (i.y+1)*(Nz/pcol)-1 }
    var rect = rect3d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, xpencil)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_xpencil_c( points  : region(ispace(int3d), coordinates),
                     xpencil : ispace(int2d) )
  var coloring = c.legion_domain_point_coloring_create()

  var prow = xpencil.bounds.hi.x + 1
  var pcol = xpencil.bounds.hi.y + 1

  var bounds = points.ispace.bounds
  var Nx = bounds.hi.x + 1
  var Ny = bounds.hi.y + 1
  var Nz = bounds.hi.z + 1

  for i in xpencil do
    var lo = int3d { x = 0, y = i.x*(Ny/prow), z = i.y*(Nz/pcol) }
    var hi = int3d { x = Nx-1, y = (i.x+1)*(Ny/prow)-1, z = (i.y+1)*(Nz/pcol)-1 }
    var rect = rect3d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, xpencil)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_ypencil( points  : region(ispace(int3d), point),
                   ypencil : ispace(int2d) )
  var coloring = c.legion_domain_point_coloring_create()

  var prow = ypencil.bounds.hi.x + 1
  var pcol = ypencil.bounds.hi.y + 1

  var bounds = points.ispace.bounds
  var Nx = bounds.hi.x + 1
  var Ny = bounds.hi.y + 1
  var Nz = bounds.hi.z + 1

  for i in ypencil do
    var lo = int3d { x = i.x*(Nx/prow), y = 0, z = i.y*(Nz/pcol) }
    var hi = int3d { x = (i.x+1)*(Nx/prow)-1, y = Ny-1, z = (i.y+1)*(Nz/pcol)-1 }
    var rect = rect3d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, ypencil)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_ypencil_c( points  : region(ispace(int3d), coordinates),
                     ypencil : ispace(int2d) )
  var coloring = c.legion_domain_point_coloring_create()

  var prow = ypencil.bounds.hi.x + 1
  var pcol = ypencil.bounds.hi.y + 1

  var bounds = points.ispace.bounds
  var Nx = bounds.hi.x + 1
  var Ny = bounds.hi.y + 1
  var Nz = bounds.hi.z + 1

  for i in ypencil do
    var lo = int3d { x = i.x*(Nx/prow), y = 0, z = i.y*(Nz/pcol) }
    var hi = int3d { x = (i.x+1)*(Nx/prow)-1, y = Ny-1, z = (i.y+1)*(Nz/pcol)-1 }
    var rect = rect3d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, ypencil)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_zpencil( points  : region(ispace(int3d), point),
                   zpencil : ispace(int2d) )
  var coloring = c.legion_domain_point_coloring_create()

  var prow = zpencil.bounds.hi.x + 1
  var pcol = zpencil.bounds.hi.y + 1

  var bounds = points.ispace.bounds
  var Nx = bounds.hi.x + 1
  var Ny = bounds.hi.y + 1
  var Nz = bounds.hi.z + 1

  for i in zpencil do
    var lo = int3d { x = i.x*(Nx/prow), y = i.y*(Ny/pcol), z = 0 }
    var hi = int3d { x = (i.x+1)*(Nx/prow)-1, y = (i.y+1)*(Ny/pcol)-1, z = Nz-1 }
    var rect = rect3d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, zpencil)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_zpencil_c( points  : region(ispace(int3d), coordinates),
                     zpencil : ispace(int2d) )
  var coloring = c.legion_domain_point_coloring_create()

  var prow = zpencil.bounds.hi.x + 1
  var pcol = zpencil.bounds.hi.y + 1

  var bounds = points.ispace.bounds
  var Nx = bounds.hi.x + 1
  var Ny = bounds.hi.y + 1
  var Nz = bounds.hi.z + 1

  for i in zpencil do
    var lo = int3d { x = i.x*(Nx/prow), y = i.y*(Ny/pcol), z = 0 }
    var hi = int3d { x = (i.x+1)*(Nx/prow)-1, y = (i.y+1)*(Ny/pcol)-1, z = Nz-1 }
    var rect = rect3d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, points, coloring, zpencil)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

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

local function make_stencil_pattern(points, index, a10, b10, c10, N, onebydx, dir, der)
  local value

  if dir == 0 then      -- x direction stencil
    if der == 1 then
      value = rexpr       - c10*points[ [poff(index, -3, 0, 0, N)] ].f end
      value = rexpr value - b10*points[ [poff(index, -2, 0, 0, N)] ].f end
      value = rexpr value - a10*points[ [poff(index, -1, 0, 0, N)] ].f end
      value = rexpr value + a10*points[ [poff(index,  1, 0, 0, N)] ].f end
      value = rexpr value + b10*points[ [poff(index,  2, 0, 0, N)] ].f end
      value = rexpr value + c10*points[ [poff(index,  3, 0, 0, N)] ].f end
      value = rexpr onebydx * ( value ) end
    elseif der == 2 then
      value = rexpr a10*( points[ [poff(index, -1, 0, 0, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 1, 0, 0, N)] ].f ) end
      value = rexpr value + b10*( points[ [poff(index, -2, 0, 0, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 2, 0, 0, N)] ].f ) end
      value = rexpr value + c10*( points[ [poff(index, -3, 0, 0, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 3, 0, 0, N)] ].f ) end
      value = rexpr onebydx*onebydx * (value) end
    end
  elseif dir == 1 then  -- y direction stencil
    if der == 1 then
      value = rexpr       - c10*points[ [poff(index, 0, -3, 0, N)] ].f end
      value = rexpr value - b10*points[ [poff(index, 0, -2, 0, N)] ].f end
      value = rexpr value - a10*points[ [poff(index, 0, -1, 0, N)] ].f end
      value = rexpr value + a10*points[ [poff(index, 0,  1, 0, N)] ].f end
      value = rexpr value + b10*points[ [poff(index, 0,  2, 0, N)] ].f end
      value = rexpr value + c10*points[ [poff(index, 0,  3, 0, N)] ].f end
      value = rexpr onebydx * ( value ) end
    elseif der == 2 then
      value = rexpr a10*( points[ [poff(index, 0, -1, 0, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 0, 1, 0, N)] ].f ) end
      value = rexpr value + b10*( points[ [poff(index, 0, -2, 0, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 0, 2, 0, N)] ].f ) end
      value = rexpr value + c10*( points[ [poff(index, 0, -3, 0, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 0, 3, 0, N)] ].f ) end
      value = rexpr onebydx*onebydx * (value) end
    end
  elseif dir == 2 then  -- z direction stencil
    if der == 1 then
      value = rexpr       - c10*points[ [poff(index, 0, 0, -3, N)] ].f end
      value = rexpr value - b10*points[ [poff(index, 0, 0, -2, N)] ].f end
      value = rexpr value - a10*points[ [poff(index, 0, 0, -1, N)] ].f end
      value = rexpr value + a10*points[ [poff(index, 0, 0,  1, N)] ].f end
      value = rexpr value + b10*points[ [poff(index, 0, 0,  2, N)] ].f end
      value = rexpr value + c10*points[ [poff(index, 0, 0,  3, N)] ].f end
      value = rexpr onebydx * ( value ) end
    elseif der == 2 then
      value = rexpr a10*( points[ [poff(index, 0, 0, -1, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 0, 0, 1, N)] ].f ) end
      value = rexpr value + b10*( points[ [poff(index, 0, 0, -2, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 0, 0, 2, N)] ].f ) end
      value = rexpr value + c10*( points[ [poff(index, 0, 0, -3, N)] ].f - 2.0*points[ index ].f + points[ [poff(index, 0, 0, 3, N)] ].f ) end
      value = rexpr onebydx*onebydx * (value) end
    end
  end
  return value
end

local function make_stencil_x(N, onebydx, a10, b10, c10, der)
  local task rhs_x( points : region(ispace(int3d), point) )
  where
    reads(points.f), writes(points.dfx)
  do
    for i in points do
      points[i].dfx = [make_stencil_pattern(points, i, a10, b10, c10, N, onebydx, 0, der)]
    end
  end
  return rhs_x
end

local function make_stencil_y(N, onebydy, a10, b10, c10, der)
  local task rhs_y( points : region(ispace(int3d), point) )
  where
    reads(points.f), writes(points.dfy)
  do
    for i in points do
      points[i].dfy = [make_stencil_pattern(points, i, a10, b10, c10, N, onebydy, 1, der)]
    end
  end
  return rhs_y
end

local function make_stencil_z(N, onebydz, a10, b10, c10, der)
  local task rhs_z( points : region(ispace(int3d), point) )
  where
    reads(points.f), writes(points.dfz)
  do
    for i in points do
      points[i].dfz = [make_stencil_pattern(points, i, a10, b10, c10, N, onebydz, 2, der)]
    end
  end
  return rhs_z
end

local ComputeXRHS  = make_stencil_x(NN, ONEBYDX, a10d1, b10d1, c10d1, 1)
local ComputeYRHS  = make_stencil_y(NN, ONEBYDY, a10d1, b10d1, c10d1, 1)
local ComputeZRHS  = make_stencil_z(NN, ONEBYDZ, a10d1, b10d1, c10d1, 1)

local ComputeX2RHS = make_stencil_x(NN, ONEBYDX, a10d2, b10d2, c10d2, 2)
local ComputeY2RHS = make_stencil_y(NN, ONEBYDY, a10d2, b10d2, c10d2, 2)
local ComputeZ2RHS = make_stencil_z(NN, ONEBYDZ, a10d2, b10d2, c10d2, 2)

task ddx( points : region(ispace(int3d), point),
          LU     : region(ispace(int1d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfx)
do
  ComputeXRHS(points)
  var token = SolveXLU(points,LU)
  return token
end

task d2dx2( points : region(ispace(int3d), point),
            LU     : region(ispace(int1d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfx)
do
  ComputeX2RHS(points)
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

task d2dy2( points : region(ispace(int3d), point),
            LU     : region(ispace(int1d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfy)
do
  ComputeY2RHS(points)
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

task d2dz2( points : region(ispace(int3d), point),
            LU     : region(ispace(int1d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfz)
do
  ComputeZ2RHS(points)
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

task get_error_d2( points : region(ispace(int3d), point) )
where
  reads(points.f, points.dfx, points.dfy, points.dfz)
do
  var err : double = 0.0
  for i in points do
    err = max(err, cmath.fabs(points[i].dfx + points[i].dfy + points[i].dfz + points[i].f))
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
  
  -- Coefficients for the 10th order 2nd derivative
  var alpha10d2 : double = 334.0/899.0
  var beta10d2  : double = 43.0/1798.0

  var grid_x = ispace(int1d, N)
  var LU_x   = region(grid_x, LU_struct)
  get_LU_decomposition(LU_x, beta10d1, alpha10d1, 1.0, alpha10d1, beta10d1)
  var LU_x2  = region(grid_x, LU_struct)
  get_LU_decomposition(LU_x2, beta10d2, alpha10d2, 1.0, alpha10d2, beta10d2)

  var grid_y = ispace(int1d, N)
  var LU_y   = region(grid_y, LU_struct)
  get_LU_decomposition(LU_y, beta10d1, alpha10d1, 1.0, alpha10d1, beta10d1)
  var LU_y2  = region(grid_y, LU_struct)
  get_LU_decomposition(LU_y2, beta10d2, alpha10d2, 1.0, alpha10d2, beta10d2)

  var grid_z = ispace(int1d, N)
  var LU_z   = region(grid_z, LU_struct)
  get_LU_decomposition(LU_z, beta10d1, alpha10d1, 1.0, alpha10d1, beta10d1)
  var LU_z2  = region(grid_z, LU_struct)
  get_LU_decomposition(LU_z2, beta10d2, alpha10d2, 1.0, alpha10d2, beta10d2)

  var grid   = ispace(int3d, { x = N, y = N, z = N })
  var coords = region(grid, coordinates)
  var points = region(grid, point)
  var exact  = region(grid, point)

  var prowcol = factorize(parallelism)

  var pencil = ispace(int2d, prowcol)
  
  var points_x = make_xpencil(points, pencil) -- Partition of x-pencils
  var points_y = make_ypencil(points, pencil) -- Partition of y-pencils
  var points_z = make_zpencil(points, pencil) -- Partition of z-pencils

  var exact_x  = make_xpencil(exact,  pencil) -- Partition of x-pencils
  var exact_y  = make_ypencil(exact,  pencil) -- Partition of y-pencils
  var exact_z  = make_zpencil(exact,  pencil) -- Partition of z-pencils

  var coords_x = make_xpencil_c(coords, pencil) -- Partition of x-pencils
  var coords_y = make_ypencil_c(coords, pencil) -- Partition of y-pencils
  var coords_z = make_zpencil_c(coords, pencil) -- Partition of z-pencils

  -- c.printf("proc 0,0 lo: {%d, %d, %d}\n", points_x[{0,0}].bounds.lo.x, points_x[{0,0}].bounds.lo.y, points_x[{0,0}].bounds.lo.z)
  -- c.printf("proc 0,0 hi: {%d, %d, %d}\n", points_x[{0,0}].bounds.hi.x, points_x[{0,0}].bounds.hi.y, points_x[{0,0}].bounds.hi.z)
  -- c.printf("proc 1,0 lo: {%d, %d, %d}\n", points_x[{1,0}].bounds.lo.x, points_x[{1,0}].bounds.lo.y, points_x[{1,0}].bounds.lo.z)
  -- c.printf("proc 1,0 hi: {%d, %d, %d}\n", points_x[{1,0}].bounds.hi.x, points_x[{1,0}].bounds.hi.y, points_x[{1,0}].bounds.hi.z)
  -- c.printf("proc 0,1 lo: {%d, %d, %d}\n", points_x[{0,1}].bounds.lo.x, points_x[{0,1}].bounds.lo.y, points_x[{0,1}].bounds.lo.z)
  -- c.printf("proc 0,1 hi: {%d, %d, %d}\n", points_x[{0,1}].bounds.hi.x, points_x[{0,1}].bounds.hi.y, points_x[{0,1}].bounds.hi.z)
  -- 
  -- c.printf("proc 0,0 lo: {%d, %d, %d}\n", points_y[{0,0}].bounds.lo.x, points_y[{0,0}].bounds.lo.y, points_y[{0,0}].bounds.lo.z)
  -- c.printf("proc 0,0 hi: {%d, %d, %d}\n", points_y[{0,0}].bounds.hi.x, points_y[{0,0}].bounds.hi.y, points_y[{0,0}].bounds.hi.z)
  -- c.printf("proc 1,0 lo: {%d, %d, %d}\n", points_y[{1,0}].bounds.lo.x, points_y[{1,0}].bounds.lo.y, points_y[{1,0}].bounds.lo.z)
  -- c.printf("proc 1,0 hi: {%d, %d, %d}\n", points_y[{1,0}].bounds.hi.x, points_y[{1,0}].bounds.hi.y, points_y[{1,0}].bounds.hi.z)
  -- c.printf("proc 0,1 lo: {%d, %d, %d}\n", points_y[{0,1}].bounds.lo.x, points_y[{0,1}].bounds.lo.y, points_y[{0,1}].bounds.lo.z)
  -- c.printf("proc 0,1 hi: {%d, %d, %d}\n", points_y[{0,1}].bounds.hi.x, points_y[{0,1}].bounds.hi.y, points_y[{0,1}].bounds.hi.z)
  -- 
  -- c.printf("proc 0,0 lo: {%d, %d, %d}\n", points_z[{0,0}].bounds.lo.x, points_z[{0,0}].bounds.lo.y, points_z[{0,0}].bounds.lo.z)
  -- c.printf("proc 0,0 hi: {%d, %d, %d}\n", points_z[{0,0}].bounds.hi.x, points_z[{0,0}].bounds.hi.y, points_z[{0,0}].bounds.hi.z)
  -- c.printf("proc 1,0 lo: {%d, %d, %d}\n", points_z[{1,0}].bounds.lo.x, points_z[{1,0}].bounds.lo.y, points_z[{1,0}].bounds.lo.z)
  -- c.printf("proc 1,0 hi: {%d, %d, %d}\n", points_z[{1,0}].bounds.hi.x, points_z[{1,0}].bounds.hi.y, points_z[{1,0}].bounds.hi.z)
  -- c.printf("proc 0,1 lo: {%d, %d, %d}\n", points_z[{0,1}].bounds.lo.x, points_z[{0,1}].bounds.lo.y, points_z[{0,1}].bounds.lo.z)
  -- c.printf("proc 0,1 hi: {%d, %d, %d}\n", points_z[{0,1}].bounds.hi.x, points_z[{0,1}].bounds.hi.y, points_z[{0,1}].bounds.hi.z)

  var token = 0 
 
  -- Initialize function f
  for i in pencil do
    token += initialize(points_x[i], exact_x[i], coords_x[i], dx, dy, dz)
  end

  wait_for(token)
  var ts_start = c.legion_get_current_time_in_micros()
  
  -- Get df/dx, df/dy, df/dz
  token += ddx(points,LU_x)
  token += ddy(points,LU_y)
  token += ddz(points,LU_z)
  
  wait_for(token)
  var ts_d1 = c.legion_get_current_time_in_micros() - ts_start
  
  var err_x = get_error_x(points,exact) 
  var err_y = get_error_y(points,exact) 
  var err_z = get_error_z(points,exact) 
  
  wait_for(err_x)
  wait_for(err_y)
  wait_for(err_z)
  ts_start = c.legion_get_current_time_in_micros()
  
  -- Get d2f/dx2, d2f/dy2, d2f/dz2
  token += d2dx2(points,LU_x2)
  token += d2dy2(points,LU_y2)
  token += d2dz2(points,LU_z2)
  
  wait_for(token)
  var ts_d2 = c.legion_get_current_time_in_micros() - ts_start
  
  var err_d2 = get_error_d2(points) 

  c.printf("Time to get the 1st derivatives: %12.5e\n", (ts_d1)*1e-6)
  c.printf("  Maximum error in x = %12.5e\n", err_x)
  c.printf("  Maximum error in y = %12.5e\n", err_y)
  c.printf("  Maximum error in z = %12.5e\n", err_z)
  c.printf("Time to get the 2nd derivatives: %12.5e\n", (ts_d2)*1e-6)
  c.printf("  Maximum error in laplacian = %12.5e\n", err_d2)

end

regentlib.start(main)
