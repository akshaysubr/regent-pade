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

local parallelism = 64

local a10d1 = ( 17.0/ 12.0)/2.0
local b10d1 = (101.0/150.0)/4.0
local c10d1 = (  1.0/100.0)/6.0

local a10d2 = (1065.0/1798.0)/1.0
local b10d2 = (1038.0/ 899.0)/4.0
local c10d2 = (  79.0/1798.0)/9.0

local gamma = 5.0/3.0
local gamma_m1 = gamma - 1.0
local onebygam_m1 = 1.0 / gamma_m1

local dt = 1.0e-5
local TSTOP = 2.0
local nsteps = math.floor(TSTOP / dt)

fspace coordinates {
  x   : double,
  y   : double,
  z   : double,
}

fspace conserved {
  rho  : double,
  rhou : double,
  rhov : double,
  rhow : double,
  rhoE : double,
}

fspace primitive {
  rho  : double,
  u    : double,
  v    : double,
  w    : double,
  p    : double,
  e    : double,
}

fspace pointx {
  f   : double,
  dfx : double,
}
fspace pointy {
  f   : double,
  dfy : double,
}
fspace pointz {
  f   : double,
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

task partitionLU( LU     : region(ispace(int3d), LU_struct),
                  pencil : ispace(int2d) )
  var coloring = c.legion_domain_point_coloring_create()

  var prow = pencil.bounds.hi.x + 1
  var pcol = pencil.bounds.hi.y + 1

  var bounds = LU.ispace.bounds
  var N = bounds.hi.x + 1

  for i in pencil do
    var lo = int3d { x = 0,   y = i.x, z = i.y }
    var hi = int3d { x = N-1, y = i.x, z = i.y }
    var rect = rect3d { lo = lo, hi = hi }
    c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  end
  var p = partition(disjoint, LU, coloring, pencil)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task make_xpencil_prim( points  : region(ispace(int3d), primitive),
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

task make_xpencil_cnsr( points  : region(ispace(int3d), conserved),
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

task make_xpencil_F( points  : region(ispace(int3d), pointx),
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

task make_ypencil_prim( points  : region(ispace(int3d), primitive),
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

task make_ypencil_cnsr( points  : region(ispace(int3d), conserved),
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

task make_ypencil_F( points  : region(ispace(int3d), pointy),
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

task make_zpencil_prim( points  : region(ispace(int3d), primitive),
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

task make_zpencil_cnsr( points  : region(ispace(int3d), conserved),
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

task make_zpencil_F( points  : region(ispace(int3d), pointz),
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

task get_LU_decomposition(LU : region(ispace(int3d), LU_struct),
                          e  : double,
                          a  : double,
                          d  : double,
                          cc : double,
                          f  : double)
where
  reads writes( LU )
do

  var N : int64 = LU.ispace.bounds.hi.x + 1
  var pr = LU.ispace.bounds.hi.y
  var pc = LU.ispace.bounds.hi.z

  -- Step 1
  LU[{0,pr,pc}].g = d
  LU[{1,pr,pc}].b = a/LU[{0,pr,pc}].g
  LU[{0,pr,pc}].h = cc
  LU[{0,pr,pc}].k = f/LU[{0,pr,pc}].g
  LU[{0,pr,pc}].w = a
  LU[{0,pr,pc}].v = e
  LU[{0,pr,pc}].l = cc/LU[{0,pr,pc}].g
  
  LU[{1,pr,pc}].g = d - LU[{1,pr,pc}].b*LU[{0,pr,pc}].h
  LU[{1,pr,pc}].k = -LU[{0,pr,pc}].k*LU[{0,pr,pc}].h/LU[{1,pr,pc}].g
  LU[{1,pr,pc}].w = e - LU[{1,pr,pc}].b*LU[{0,pr,pc}].w
  LU[{1,pr,pc}].v = -LU[{1,pr,pc}].b*LU[{0,pr,pc}].v
  LU[{1,pr,pc}].l = (f - LU[{0,pr,pc}].l*LU[{0,pr,pc}].h) / LU[{1,pr,pc}].g
  LU[{1,pr,pc}].h = cc - LU[{1,pr,pc}].b*f

  -- Step 2
  for i = 2,N-3 do
    LU[{i,pr,pc}].b = ( a - ( e/LU[{i-2,pr,pc}].g )*LU[{i-2,pr,pc}].h ) / LU[{i-1,pr,pc}].g
    LU[{i,pr,pc}].h = cc - LU[{i,pr,pc}].b*f
    LU[{i,pr,pc}].g = d - ( e/LU[{i-2,pr,pc}].g )*f - LU[{i,pr,pc}].b*LU[{i-1,pr,pc}].h
  end

  -- Step 3
  LU[{N-3,pr,pc}].b = ( a - ( e/LU[{N-5,pr,pc}].g )*LU[{N-5,pr,pc}].h ) / LU[{N-4,pr,pc}].g
  LU[{N-3,pr,pc}].g = d - ( e/LU[{N-5,pr,pc}].g )*f - LU[{N-3,pr,pc}].b*LU[{N-4,pr,pc}].h

  -- Step 4
  for i = 2,N-4 do
    LU[{i,pr,pc}].k = -( LU[{i-2,pr,pc}].k*f + LU[{i-1,pr,pc}].k*LU[{i-1,pr,pc}].h ) / LU[{i,pr,pc}].g
    LU[{i,pr,pc}].v = -( e/LU[{i-2,pr,pc}].g )*LU[{i-2,pr,pc}].v - LU[{i,pr,pc}].b*LU[{i-1,pr,pc}].v
  end

  -- Step 5
  LU[{N-4,pr,pc}].k = ( e - LU[{N-6,pr,pc}].k*f - LU[{N-5,pr,pc}].k*LU[{N-5,pr,pc}].h ) / LU[{N-4,pr,pc}].g
  LU[{N-3,pr,pc}].k = ( a - LU[{N-5,pr,pc}].k*f - LU[{N-4,pr,pc}].k*LU[{N-4,pr,pc}].h ) / LU[{N-3,pr,pc}].g
  LU[{N-4,pr,pc}].v = f  - ( e/LU[{N-6,pr,pc}].g )*LU[{N-6,pr,pc}].v - LU[{N-4,pr,pc}].b*LU[{N-5,pr,pc}].v
  LU[{N-3,pr,pc}].v = cc - ( e/LU[{N-5,pr,pc}].g )*LU[{N-5,pr,pc}].v - LU[{N-3,pr,pc}].b*LU[{N-4,pr,pc}].v
  LU[{N-2,pr,pc}].g = d
  for i = 0,N-2 do
    LU[{N-2,pr,pc}].g -= LU[{i,pr,pc}].k*LU[{i,pr,pc}].v
  end

  -- Step 6
  for i = 2,N-3 do
    LU[{i,pr,pc}].w = -( e/LU[{i-2,pr,pc}].g )*LU[{i-2,pr,pc}].w - LU[{i,pr,pc}].b*LU[{i-1,pr,pc}].w
    LU[{i,pr,pc}].l = -( LU[{i-2,pr,pc}].l*f + LU[{i-1,pr,pc}].l*LU[{i-1,pr,pc}].h ) / LU[{i,pr,pc}].g
  end

  -- Step 7
  LU[{N-3,pr,pc}].w = f - ( e/LU[{N-5,pr,pc}].g )*LU[{N-5,pr,pc}].w - LU[{N-3,pr,pc}].b*LU[{N-4,pr,pc}].w
  LU[{N-2,pr,pc}].w = cc
  for i = 0,N-2 do
    LU[{N-2,pr,pc}].w -= LU[{i,pr,pc}].k*LU[{i,pr,pc}].w
  end
  LU[{N-3,pr,pc}].l = ( e - LU[{N-5,pr,pc}].l*f - LU[{N-4,pr,pc}].l*LU[{N-4,pr,pc}].h ) / LU[{N-3,pr,pc}].g
  LU[{N-2,pr,pc}].l = a
  for i = 0,N-2 do
    LU[{N-2,pr,pc}].l -= LU[{i,pr,pc}].l*LU[{i,pr,pc}].v
  end
  LU[{N-2,pr,pc}].l = LU[{N-2,pr,pc}].l / LU[{N-2,pr,pc}].g
  LU[{N-1,pr,pc}].g = d
  for i = 0,N-1 do
    LU[{N-1,pr,pc}].g -= LU[{i,pr,pc}].l*LU[{i,pr,pc}].w
  end

  -- Set eg = e/g
  for i = 2,N-2 do
    LU[{i,pr,pc}].eg = e/LU[{i-2,pr,pc}].g
  end

  -- Set ff = f
  for i = 0,N-4 do
    LU[{i,pr,pc}].ff = f
  end

  -- Set g = 1/g
  for i = 0,N do
    LU[{i,pr,pc}].g = 1.0/LU[{i,pr,pc}].g
  end

  -- c.printf("LU decomposition:\n")
  -- for i = 0,N do
  --   c.printf("%8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f, %8.5f\n",LU[i].b,LU[i].eg,LU[i].k,LU[i].l,LU[i].g,LU[i].h,LU[i].ff,LU[i].v,LU[i].w)
  -- end

end

task SolveXLU( points : region(ispace(int3d), pointx),
               LU     : region(ispace(int3d), LU_struct) )
where
  reads writes(points.dfx), reads(LU)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.x + 1
  var pr = LU.ispace.bounds.hi.y
  var pc = LU.ispace.bounds.hi.z

  for j = bounds.lo.y, bounds.hi.y+1 do
    for k = bounds.lo.z, bounds.hi.z+1 do

      -- Step 8
      points[{1,j,k}].dfx = points[{1,j,k}].dfx - LU[{1,pr,pc}].b*points[{0,j,k}].dfx
      var sum1 : double = LU[{0,pr,pc}].k*points[{0,j,k}].dfx + LU[{1,pr,pc}].k*points[{1,j,k}].dfx
      var sum2 : double = LU[{0,pr,pc}].l*points[{0,j,k}].dfx + LU[{1,pr,pc}].l*points[{1,j,k}].dfx

      -- Step 9
      for i = 2,N-2 do
        points[{i,j,k}].dfx = points[{i,j,k}].dfx - LU[{i,pr,pc}].b*points[{i-1,j,k}].dfx - LU[{i,pr,pc}].eg*points[{i-2,j,k}].dfx
        sum1 += LU[{i,pr,pc}].k*points[{i,j,k}].dfx
        sum2 += LU[{i,pr,pc}].l*points[{i,j,k}].dfx
      end

      -- Step 10
      points[{N-2,j,k}].dfx = points[{N-2,j,k}].dfx - sum1
      points[{N-1,j,k}].dfx = ( points[{N-1,j,k}].dfx - sum2 - LU[{N-2,pr,pc}].l*points[{N-2,j,k}].dfx )*LU[{N-1,pr,pc}].g

      -- Step 11
      points[{N-2,j,k}].dfx = ( points[{N-2,j,k}].dfx - LU[{N-2,pr,pc}].w*points[{N-1,j,k}].dfx )*LU[{N-2,pr,pc}].g
      points[{N-3,j,k}].dfx = ( points[{N-3,j,k}].dfx - LU[{N-3,pr,pc}].v*points[{N-2,j,k}].dfx - LU[{N-3,pr,pc}].w*points[{N-1,j,k}].dfx )*LU[{N-3,pr,pc}].g
      points[{N-4,j,k}].dfx = ( points[{N-4,j,k}].dfx - LU[{N-4,pr,pc}].h*points[{N-3,j,k}].dfx - LU[{N-4,pr,pc}].v*points[{N-2,j,k}].dfx - LU[{N-4,pr,pc}].w*points[{N-1,j,k}].dfx )*LU[{N-4,pr,pc}].g
      for i = N-5,-1,-1 do
        points[{i,j,k}].dfx = ( points[{i,j,k}].dfx - LU[{i,pr,pc}].h*points[{i+1,j,k}].dfx - LU[{i,pr,pc}].ff*points[{i+2,j,k}].dfx - LU[{i,pr,pc}].v*points[{N-2,j,k}].dfx - LU[{i,pr,pc}].w*points[{N-1,j,k}].dfx )*LU[{i,pr,pc}].g
      end

    end
  end
  return 1
end

task SolveYLU( points : region(ispace(int3d), pointy),
               LU     : region(ispace(int3d), LU_struct) )
where
  reads writes(points.dfy), reads(LU)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.y + 1
  var pr = LU.ispace.bounds.hi.y
  var pc = LU.ispace.bounds.hi.z

  for i = bounds.lo.x, bounds.hi.x+1 do
    for k = bounds.lo.z, bounds.hi.z+1 do

      -- Step 8
      points[{i,1,k}].dfy = points[{i,1,k}].dfy - LU[{1,pr,pc}].b*points[{i,0,k}].dfy
      var sum1 : double = LU[{0,pr,pc}].k*points[{i,0,k}].dfy + LU[{1,pr,pc}].k*points[{i,1,k}].dfy
      var sum2 : double = LU[{0,pr,pc}].l*points[{i,0,k}].dfy + LU[{1,pr,pc}].l*points[{i,1,k}].dfy

      -- Step 9
      for j = 2,N-2 do
        points[{i,j,k}].dfy = points[{i,j,k}].dfy - LU[{j,pr,pc}].b*points[{i,j-1,k}].dfy - LU[{j,pr,pc}].eg*points[{i,j-2,k}].dfy
        sum1 += LU[{j,pr,pc}].k*points[{i,j,k}].dfy
        sum2 += LU[{j,pr,pc}].l*points[{i,j,k}].dfy
      end

      -- Step 10
      points[{i,N-2,k}].dfy = points[{i,N-2,k}].dfy - sum1
      points[{i,N-1,k}].dfy = ( points[{i,N-1,k}].dfy - sum2 - LU[{N-2,pr,pc}].l*points[{i,N-2,k}].dfy )*LU[{N-1,pr,pc}].g

      -- Step 11
      points[{i,N-2,k}].dfy = ( points[{i,N-2,k}].dfy - LU[{N-2,pr,pc}].w*points[{i,N-1,k}].dfy )*LU[{N-2,pr,pc}].g
      points[{i,N-3,k}].dfy = ( points[{i,N-3,k}].dfy - LU[{N-3,pr,pc}].v*points[{i,N-2,k}].dfy - LU[{N-3,pr,pc}].w*points[{i,N-1,k}].dfy )*LU[{N-3,pr,pc}].g
      points[{i,N-4,k}].dfy = ( points[{i,N-4,k}].dfy - LU[{N-4,pr,pc}].h*points[{i,N-3,k}].dfy - LU[{N-4,pr,pc}].v*points[{i,N-2,k}].dfy - LU[{N-4,pr,pc}].w*points[{i,N-1,k}].dfy )*LU[{N-4,pr,pc}].g
      for j = N-5,-1,-1 do
        points[{i,j,k}].dfy = ( points[{i,j,k}].dfy - LU[{j,pr,pc}].h*points[{i,j+1,k}].dfy - LU[{j,pr,pc}].ff*points[{i,j+2,k}].dfy - LU[{j,pr,pc}].v*points[{i,N-2,k}].dfy - LU[{j,pr,pc}].w*points[{i,N-1,k}].dfy )*LU[{j,pr,pc}].g
      end

    end
  end

  return 1
end

task SolveZLU( points : region(ispace(int3d), pointz),
               LU     : region(ispace(int3d), LU_struct) )
where
  reads writes(points.dfz), reads(LU)
do
  var bounds = points.ispace.bounds
  var N = bounds.hi.z + 1
  var pr = LU.ispace.bounds.hi.y
  var pc = LU.ispace.bounds.hi.z

  for i = bounds.lo.x, bounds.hi.x+1 do
    for j = bounds.lo.y, bounds.hi.y+1 do

      -- Step 8
      points[{i,j,1}].dfz = points[{i,j,1}].dfz - LU[{1,pr,pc}].b*points[{i,j,0}].dfz
      var sum1 : double = LU[{0,pr,pc}].k*points[{i,j,0}].dfz + LU[{1,pr,pc}].k*points[{i,j,1}].dfz
      var sum2 : double = LU[{0,pr,pc}].l*points[{i,j,0}].dfz + LU[{1,pr,pc}].l*points[{i,j,1}].dfz

      -- Step 9
      for k = 2,N-2 do
        points[{i,j,k}].dfz = points[{i,j,k}].dfz - LU[{k,pr,pc}].b*points[{i,j,k-1}].dfz - LU[{k,pr,pc}].eg*points[{i,j,k-2}].dfz
        sum1 += LU[{k,pr,pc}].k*points[{i,j,k}].dfz
        sum2 += LU[{k,pr,pc}].l*points[{i,j,k}].dfz
      end

      -- Step 10
      points[{i,j,N-2}].dfz = points[{i,j,N-2}].dfz - sum1
      points[{i,j,N-1}].dfz = ( points[{i,j,N-1}].dfz - sum2 - LU[{N-2,pr,pc}].l*points[{i,j,N-2}].dfz )*LU[{N-1,pr,pc}].g

      -- Step 11
      points[{i,j,N-2}].dfz = ( points[{i,j,N-2}].dfz - LU[{N-2,pr,pc}].w*points[{i,j,N-1}].dfz )*LU[{N-2,pr,pc}].g
      points[{i,j,N-3}].dfz = ( points[{i,j,N-3}].dfz - LU[{N-3,pr,pc}].v*points[{i,j,N-2}].dfz - LU[{N-3,pr,pc}].w*points[{i,j,N-1}].dfz )*LU[{N-3,pr,pc}].g
      points[{i,j,N-4}].dfz = ( points[{i,j,N-4}].dfz - LU[{N-4,pr,pc}].h*points[{i,j,N-3}].dfz - LU[{N-4,pr,pc}].v*points[{i,j,N-2}].dfz - LU[{N-4,pr,pc}].w*points[{i,j,N-1}].dfz )*LU[{N-4,pr,pc}].g
      for k = N-5,-1,-1 do
        points[{i,j,k}].dfz = ( points[{i,j,k}].dfz - LU[{k,pr,pc}].h*points[{i,j,k+1}].dfz - LU[{k,pr,pc}].ff*points[{i,j,k+2}].dfz - LU[{k,pr,pc}].v*points[{i,j,N-2}].dfz - LU[{k,pr,pc}].w*points[{i,j,N-1}].dfz )*LU[{k,pr,pc}].g
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
  local task rhs_x( points : region(ispace(int3d), pointx) )
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
  local task rhs_y( points : region(ispace(int3d), pointy) )
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
  local task rhs_z( points : region(ispace(int3d), pointz) )
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

task ddx( points : region(ispace(int3d), pointx),
          LU     : region(ispace(int3d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfx)
do
  ComputeXRHS(points)
  var token = SolveXLU(points,LU)
  --c.printf("In ddx\n")
  --for p in points do
  --  if p.x == 0 and p.y == 0 and p.z == 0 then
  --    c.printf("{%d,%d,%d}: %8.5f, %8.5f\n",p.x,p.y,p.z,points[p].f,points[p].dfx)
  --  end
  --end
  token = points[points.ispace.bounds.lo].dfx
  return token
end

task d2dx2( points : region(ispace(int3d), pointx),
            LU     : region(ispace(int3d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfx)
do
  ComputeX2RHS(points)
  var token = SolveXLU(points,LU)
  --c.printf("In d2dx2\n")
  --for p in points do
  --  if p.x == 0 and p.y == 0 and p.z == 0 then
  --    c.printf("{%d,%d,%d}: %8.5f, %8.5f\n",p.x,p.y,p.z,points[p].f,points[p].dfx)
  --  end
  --end
  token = points[points.ispace.bounds.lo].dfx
  return token
end

task ddy( points : region(ispace(int3d), pointy),
          LU     : region(ispace(int3d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfy)
do
  ComputeYRHS(points)
  var token = SolveYLU(points,LU)
  --c.printf("In ddy\n")
  --for p in points do
  --  if p.x == 0 and p.y == 0 and p.z == 0 then
  --    c.printf("{%d,%d,%d}: %8.5f, %8.5f\n",p.x,p.y,p.z,points[p].f,points[p].dfy)
  --  end
  --end
  token = points[points.ispace.bounds.lo].dfy
  return token
end

task d2dy2( points : region(ispace(int3d), pointy),
            LU     : region(ispace(int3d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfy)
do
  ComputeY2RHS(points)
  var token = SolveYLU(points,LU)
  --c.printf("In d2dy2\n")
  --for p in points do
  --  if p.x == 0 and p.y == 0 and p.z == 0 then
  --    c.printf("{%d,%d,%d}: %8.5f, %8.5f\n",p.x,p.y,p.z,points[p].f,points[p].dfy)
  --  end
  --end
  token = points[points.ispace.bounds.lo].dfy
  return token
end

task ddz( points : region(ispace(int3d), pointz),
          LU     : region(ispace(int3d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfz)
do
  ComputeZRHS(points)
  var token = SolveZLU(points,LU)
  --c.printf("In ddz\n")
  --for p in points do
  --  if p.x == 0 and p.y == 0 and p.z == 0 then
  --    c.printf("{%d,%d,%d}: %8.5f, %8.5f\n",p.x,p.y,p.z,points[p].f,points[p].dfz)
  --  end
  --end
  token = points[points.ispace.bounds.lo].dfz
  return token
end

task d2dz2( points : region(ispace(int3d), pointz),
            LU     : region(ispace(int3d), LU_struct) )
where
  reads(LU, points.f), reads writes(points.dfz)
do
  ComputeZ2RHS(points)
  var token = SolveZLU(points,LU)
  --c.printf("In d2dz2\n")
  --for p in points do
  --  if p.x == 0 and p.y == 0 and p.z == 0 then
  --    c.printf("{%d,%d,%d}: %8.5f, %8.5f\n",p.x,p.y,p.z,points[p].f,points[p].dfz)
  --  end
  --end
  token = points[points.ispace.bounds.lo].dfz
  return token
end

-- task gradient( points_x : region(ispace(int3d), point),
--                points_y : region(ispace(int3d), point),
--                points_z : region(ispace(int3d), point),
--                LU_x     : region(ispace(int3d), LU_struct),
--                LU_y     : region(ispace(int3d), LU_struct),
--                LU_z     : region(ispace(int3d), LU_struct) )
-- where
--   reads(LU_x, LU_y, LU_z), reads(points_x.f, points_y.f, points_z.f),
--   reads writes(points_x.dfx, points_y.dfy, points_z.dfz)
-- do 
--   var token = 0
--   token += ddx(points_x,LU_x)
--   token += ddy(points_y,LU_y)
--   token += ddz(points_z,LU_z)
--   return token
-- end

task get_e_from_p( Wprim : region(ispace(int3d), primitive) )
where
  reads (Wprim.p, Wprim.rho), reads writes(Wprim.e)
do
  for i in Wprim do
    Wprim[i].e = Wprim[i].p * onebygam_m1 / Wprim[i].rho
  end
end

task get_conserved( Wprim : region(ispace(int3d), primitive),
                    Wcnsr : region(ispace(int3d), conserved) )
where
  reads(Wprim.rho,Wprim.u,Wprim.v,Wprim.w,Wprim.p), reads writes(Wcnsr, Wprim.e)
do

  get_e_from_p(Wprim)

  for i in Wcnsr do
    Wcnsr[i].rho  = Wprim[i].rho
    Wcnsr[i].rhou = Wprim[i].rho * Wprim[i].u
    Wcnsr[i].rhov = Wprim[i].rho * Wprim[i].v
    Wcnsr[i].rhow = Wprim[i].rho * Wprim[i].w
    Wcnsr[i].rhoE = Wprim[i].rho * ( Wprim[i].e + 0.5*(Wprim[i].u*Wprim[i].u+Wprim[i].v*Wprim[i].v+Wprim[i].w*Wprim[i].w) )
  end
end

task get_primitive( Wcnsr : region(ispace(int3d), conserved),
                    Wprim : region(ispace(int3d), primitive) )
where
  reads(Wcnsr), reads writes(Wprim)
do

  for i in Wprim do
    Wprim[i].rho = Wcnsr[i].rho
    Wprim[i].u   = Wcnsr[i].rhou / Wcnsr[i].rho
    Wprim[i].v   = Wcnsr[i].rhov / Wcnsr[i].rho
    Wprim[i].w   = Wcnsr[i].rhow / Wcnsr[i].rho
    Wprim[i].e   =(Wcnsr[i].rhoE / Wcnsr[i].rho) - 0.5*(Wprim[i].u*Wprim[i].u+Wprim[i].v*Wprim[i].v+Wprim[i].w*Wprim[i].w)
    Wprim[i].p   = gamma_m1 * Wprim[i].rho * Wprim[i].e
  end
end

task get_xfluxes( Wprim : region(ispace(int3d), primitive),
                  Wcnsr : region(ispace(int3d), conserved),
                  FX0   : region(ispace(int3d), pointx),
                  FX1   : region(ispace(int3d), pointx),
                  FX2   : region(ispace(int3d), pointx),
                  FX3   : region(ispace(int3d), pointx),
                  FX4   : region(ispace(int3d), pointx) )
where
  reads (Wprim, Wcnsr), reads writes (FX0.f, FX1.f, FX2.f, FX3.f, FX4.f)
do
  for i in Wprim do
    FX0[i].f = Wcnsr[i].rhou
    FX1[i].f = Wcnsr[i].rhou * Wprim[i].u + Wprim[i].p
    FX2[i].f = Wcnsr[i].rhou * Wprim[i].v
    FX3[i].f = Wcnsr[i].rhou * Wprim[i].w
    FX4[i].f =(Wcnsr[i].rhoE + Wprim[i].p) * Wprim[i].u
  end
end

task get_yfluxes( Wprim : region(ispace(int3d), primitive),
                  Wcnsr : region(ispace(int3d), conserved),
                  FY0   : region(ispace(int3d), pointy),
                  FY1   : region(ispace(int3d), pointy),
                  FY2   : region(ispace(int3d), pointy),
                  FY3   : region(ispace(int3d), pointy),
                  FY4   : region(ispace(int3d), pointy) )
where
  reads (Wprim, Wcnsr), reads writes (FY0.f, FY1.f, FY2.f, FY3.f, FY4.f)
do
  for i in Wprim do
    FY0[i].f = Wcnsr[i].rhov
    FY1[i].f = Wcnsr[i].rhov * Wprim[i].u
    FY2[i].f = Wcnsr[i].rhov * Wprim[i].v + Wprim[i].p
    FY3[i].f = Wcnsr[i].rhov * Wprim[i].w
    FY4[i].f =(Wcnsr[i].rhoE + Wprim[i].p) * Wprim[i].v
  end
end

task get_zfluxes( Wprim : region(ispace(int3d), primitive),
                  Wcnsr : region(ispace(int3d), conserved),
                  FZ0   : region(ispace(int3d), pointz),
                  FZ1   : region(ispace(int3d), pointz),
                  FZ2   : region(ispace(int3d), pointz),
                  FZ3   : region(ispace(int3d), pointz),
                  FZ4   : region(ispace(int3d), pointz) )
where
  reads (Wprim, Wcnsr), reads writes (FZ0.f, FZ1.f, FZ2.f, FZ3.f, FZ4.f)
do
  for i in Wprim do
    FZ0[i].f = Wcnsr[i].rhow
    FZ1[i].f = Wcnsr[i].rhow * Wprim[i].u
    FZ2[i].f = Wcnsr[i].rhow * Wprim[i].v
    FZ3[i].f = Wcnsr[i].rhow * Wprim[i].w + Wprim[i].p
    FZ4[i].f =(Wcnsr[i].rhoE + Wprim[i].p) * Wprim[i].w
  end
end

task set_rhs_zero( rhs : region(ispace(int3d), conserved) )
where
  reads writes(rhs)
do
  for i in rhs do
    rhs[i].rho  = 0.0
    rhs[i].rhou = 0.0
    rhs[i].rhov = 0.0
    rhs[i].rhow = 0.0
    rhs[i].rhoE = 0.0
  end
end

task add_xfluxes_to_rhs( FX0 : region(ispace(int3d), pointx),
                         FX1 : region(ispace(int3d), pointx),
                         FX2 : region(ispace(int3d), pointx),
                         FX3 : region(ispace(int3d), pointx),
                         FX4 : region(ispace(int3d), pointx),
                         rhs : region(ispace(int3d), conserved),
                         dt  : double )
where
  reads(FX0.dfx, FX1.dfx, FX2.dfx, FX3.dfx, FX4.dfx), reads writes(rhs)
do
  for i in rhs do
    rhs[i].rho  -= dt*FX0[i].dfx
    rhs[i].rhou -= dt*FX1[i].dfx
    rhs[i].rhov -= dt*FX2[i].dfx
    rhs[i].rhow -= dt*FX3[i].dfx
    rhs[i].rhoE -= dt*FX4[i].dfx
  end
end

task add_yfluxes_to_rhs( FY0 : region(ispace(int3d), pointy),
                         FY1 : region(ispace(int3d), pointy),
                         FY2 : region(ispace(int3d), pointy),
                         FY3 : region(ispace(int3d), pointy),
                         FY4 : region(ispace(int3d), pointy),
                         rhs : region(ispace(int3d), conserved),
                         dt  : double )
where
  reads(FY0.dfy, FY1.dfy, FY2.dfy, FY3.dfy, FY4.dfy), reads writes(rhs)
do
  for i in rhs do
    rhs[i].rho  -= dt*FY0[i].dfy
    rhs[i].rhou -= dt*FY1[i].dfy
    rhs[i].rhov -= dt*FY2[i].dfy
    rhs[i].rhow -= dt*FY3[i].dfy
    rhs[i].rhoE -= dt*FY4[i].dfy
  end
end

task add_zfluxes_to_rhs( FZ0 : region(ispace(int3d), pointz),
                         FZ1 : region(ispace(int3d), pointz),
                         FZ2 : region(ispace(int3d), pointz),
                         FZ3 : region(ispace(int3d), pointz),
                         FZ4 : region(ispace(int3d), pointz),
                         rhs : region(ispace(int3d), conserved),
                         dt  : double )
where
  reads(FZ0.dfz, FZ1.dfz, FZ2.dfz, FZ3.dfz, FZ4.dfz), reads writes(rhs)
do
  for i in rhs do
    rhs[i].rho  -= dt*FZ0[i].dfz
    rhs[i].rhou -= dt*FZ1[i].dfz
    rhs[i].rhov -= dt*FZ2[i].dfz
    rhs[i].rhow -= dt*FZ3[i].dfz
    rhs[i].rhoE -= dt*FZ4[i].dfz
  end
end

task update_conserved( Wcnsr : region(ispace(int3d), conserved),
                       rhs   : region(ispace(int3d), conserved) )
where
  reads(rhs), reads writes(Wcnsr)
do
  for i in Wcnsr do
    Wcnsr[i].rho  += rhs[i].rho 
    Wcnsr[i].rhou += rhs[i].rhou
    Wcnsr[i].rhov += rhs[i].rhov
    Wcnsr[i].rhow += rhs[i].rhow
    Wcnsr[i].rhoE += rhs[i].rhoE
  end
end

task get_TKE( Wprim : region(ispace(int3d), primitive) )
where
  reads(Wprim.rho, Wprim.u, Wprim.v, Wprim.w)
do
  var tke : double = 0.0
  for i in Wprim do
    tke += 0.5 * Wprim[i].rho * ( Wprim[i].u*Wprim[i].u + Wprim[i].v*Wprim[i].v + Wprim[i].w*Wprim[i].w )
  end
  return tke
end

task initialize( Wprim  : region(ispace(int3d), primitive),
                 coords : region(ispace(int3d), coordinates),
                 dx     : double,
                 dy     : double,
                 dz     : double )
where
  reads writes(coords.x, coords.y, coords.z, Wprim.rho, Wprim.u, Wprim.v, Wprim.w, Wprim.p)
do
  var bounds = coords.ispace.bounds

  for i = bounds.lo.x, bounds.hi.x+1 do
    for j = bounds.lo.y, bounds.hi.y+1 do
      for k = bounds.lo.z, bounds.hi.z+1 do
        var e : int3d = { x = i, y = j, z = k }
        coords[e].x   = i*dx
        coords[e].y   = j*dy
        coords[e].z   = k*dz

        Wprim [e].rho = 1.0
        Wprim [e].u   = cmath.sin(coords[e].x) * cmath.cos(coords[e].y) * cmath.cos(coords[e].z)
        Wprim [e].v   =-cmath.cos(coords[e].x) * cmath.sin(coords[e].y) * cmath.cos(coords[e].z)
        Wprim [e].w   = 0.0
        Wprim [e].p   = 100.0 + ( (cmath.cos(2.0*coords[e].z)+2.0)*(cmath.cos(2.0*coords[e].x) + cmath.cos(2.0*coords[e].y)) - 2.0 )/16.0

      end
    end
  end
 
  return 0
end

task block_task( Wcnsr : region(ispace(int3d), conserved), 
                 Wprim : region(ispace(int3d), primitive) )
where
 reads writes(Wcnsr, Wprim)
do
 return 1
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
  c.printf("                   dt = %f\n", dt)
  c.printf("          parallelism = %d\n", parallelism)
  c.printf("====================================================\n")

  -- Coefficients for the 10th order 1st derivative
  var alpha10d1 : double = 1.0/2.0
  var beta10d1  : double = 1.0/20.0
  
  -- Coefficients for the 10th order 2nd derivative
  var alpha10d2 : double = 334.0/899.0
  var beta10d2  : double = 43.0/1798.0

  var prowcol = factorize(parallelism)
  var pencil = ispace(int2d, prowcol)
  
  var grid_x = ispace(int3d, { x = N, y = prowcol.x, z = prowcol.y } )
  var LU_x   = region(grid_x, LU_struct)
  var LU_x2  = region(grid_x, LU_struct)
  
  var pLU_x  = partitionLU(LU_x,  pencil)
  var pLU_x2 = partitionLU(LU_x2, pencil)
  must_epoch
    __demand(__parallel)
    for i in pencil do
      get_LU_decomposition(pLU_x [i], beta10d1, alpha10d1, 1.0, alpha10d1, beta10d1)
    end
  end
  must_epoch
    __demand(__parallel)
    for i in pencil do
      get_LU_decomposition(pLU_x2[i], beta10d2, alpha10d2, 1.0, alpha10d2, beta10d2)
    end
  end

  var grid_y = ispace(int3d, { x = N, y = prowcol.x, z = prowcol.y } )
  var LU_y   = region(grid_y, LU_struct)
  var LU_y2  = region(grid_y, LU_struct)

  var pLU_y  = partitionLU(LU_y,  pencil)
  var pLU_y2 = partitionLU(LU_y2, pencil)
  must_epoch
    __demand(__parallel)
    for i in pencil do
      get_LU_decomposition(pLU_y [i], beta10d1, alpha10d1, 1.0, alpha10d1, beta10d1)
    end
  end
  must_epoch
    __demand(__parallel)
    for i in pencil do
      get_LU_decomposition(pLU_y2[i], beta10d2, alpha10d2, 1.0, alpha10d2, beta10d2)
    end
  end

  var grid_z = ispace(int3d, { x = N, y = prowcol.x, z = prowcol.y } )
  var LU_z   = region(grid_z, LU_struct)
  var LU_z2  = region(grid_z, LU_struct)

  var pLU_z  = partitionLU(LU_z,  pencil)
  var pLU_z2 = partitionLU(LU_z2, pencil)  
  must_epoch
    __demand(__parallel)
    for i in pencil do
      get_LU_decomposition(pLU_z [i], beta10d1, alpha10d1, 1.0, alpha10d1, beta10d1)
    end
  end
  must_epoch
    __demand(__parallel)
    for i in pencil do
      get_LU_decomposition(pLU_z2[i], beta10d2, alpha10d2, 1.0, alpha10d2, beta10d2)
    end
  end

  var grid   = ispace(int3d, { x = N, y = N, z = N })
  var coords = region(grid, coordinates)
  var Wprim  = region(grid, primitive)
  var Wcnsr  = region(grid, conserved)
  var rhs    = region(grid, conserved)
  
  var Wprim_x = make_xpencil_prim(Wprim, pencil) -- Partition of x-pencils
  var Wprim_y = make_ypencil_prim(Wprim, pencil) -- Partition of y-pencils
  var Wprim_z = make_zpencil_prim(Wprim, pencil) -- Partition of z-pencils

  var Wcnsr_x = make_xpencil_cnsr(Wcnsr, pencil) -- Partition of x-pencils
  var Wcnsr_y = make_ypencil_cnsr(Wcnsr, pencil) -- Partition of y-pencils
  var Wcnsr_z = make_zpencil_cnsr(Wcnsr, pencil) -- Partition of z-pencils

  var rhs_x = make_xpencil_cnsr(rhs, pencil) -- Partition of x-pencils
  var rhs_y = make_ypencil_cnsr(rhs, pencil) -- Partition of y-pencils
  var rhs_z = make_zpencil_cnsr(rhs, pencil) -- Partition of z-pencils

  var coords_x = make_xpencil_c(coords, pencil) -- Partition of x-pencils
  var coords_y = make_ypencil_c(coords, pencil) -- Partition of y-pencils
  var coords_z = make_zpencil_c(coords, pencil) -- Partition of z-pencils
 
  var FX0  = region(grid, pointx)
  var FX1  = region(grid, pointx)
  var FX2  = region(grid, pointx)
  var FX3  = region(grid, pointx)
  var FX4  = region(grid, pointx)
  var FX0_x = make_xpencil_F(FX0, pencil) -- Partition of x-pencils
  var FX1_x = make_xpencil_F(FX1, pencil) -- Partition of x-pencils
  var FX2_x = make_xpencil_F(FX2, pencil) -- Partition of x-pencils
  var FX3_x = make_xpencil_F(FX3, pencil) -- Partition of x-pencils
  var FX4_x = make_xpencil_F(FX4, pencil) -- Partition of x-pencils

  var FY0  = region(grid, pointy)
  var FY1  = region(grid, pointy)
  var FY2  = region(grid, pointy)
  var FY3  = region(grid, pointy)
  var FY4  = region(grid, pointy)
  var FY0_y = make_ypencil_F(FY0, pencil) -- Partition of y-pencils
  var FY1_y = make_ypencil_F(FY1, pencil) -- Partition of y-pencils
  var FY2_y = make_ypencil_F(FY2, pencil) -- Partition of y-pencils
  var FY3_y = make_ypencil_F(FY3, pencil) -- Partition of y-pencils
  var FY4_y = make_ypencil_F(FY4, pencil) -- Partition of y-pencils

  var FZ0  = region(grid, pointz)
  var FZ1  = region(grid, pointz)
  var FZ2  = region(grid, pointz)
  var FZ3  = region(grid, pointz)
  var FZ4  = region(grid, pointz)
  var FZ0_z = make_zpencil_F(FZ0, pencil) -- Partition of z-pencils
  var FZ1_z = make_zpencil_F(FZ1, pencil) -- Partition of z-pencils
  var FZ2_z = make_zpencil_F(FZ2, pencil) -- Partition of z-pencils
  var FZ3_z = make_zpencil_F(FZ3, pencil) -- Partition of z-pencils
  var FZ4_z = make_zpencil_F(FZ4, pencil) -- Partition of z-pencils

  var token = 0 
  -- Initialize function f
  __demand(__parallel)
  for i in pencil do
    token += initialize(Wprim_x[i], coords_x[i], dx, dy, dz)
  end

  var t : double = 0.0
  var tstop : double = TSTOP
 
  var tke0 : double = 0.0
  __demand(__parallel)
  for i in pencil do
    tke0 += get_TKE( Wprim_x[i] )
  end
  tke0 = tke0 / (N*N*N)  -- Average TKE
  c.printf("Initial TKE = %8.5f\n", tke0)

  -- for step = 0,2 do
    __demand(__parallel)
    for i in pencil do
      token += block_task( Wcnsr_x[i], Wprim_x[i] )
    end 
    wait_for(token)
    var ts_start = c.legion_get_current_time_in_micros()

    __demand(__parallel)
    for i in pencil do
      get_conserved(Wprim_x[i], Wcnsr_x[i])
    end

    __demand(__parallel)
    for i in pencil do
      set_rhs_zero(rhs_x[i])
    end

    __demand(__parallel)
    for i in pencil do
      get_xfluxes(Wprim_x[i], Wcnsr_x[i], FX0_x[i], FX1_x[i], FX2_x[i], FX3_x[i], FX4_x[i])
    end

    __demand(__parallel)
    for i in pencil do
      ddx(FX0_x[i], pLU_x[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddx(FX1_x[i], pLU_x[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddx(FX2_x[i], pLU_x[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddx(FX3_x[i], pLU_x[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddx(FX4_x[i], pLU_x[i])
    end

    __demand(__parallel)
    for i in pencil do
      add_xfluxes_to_rhs(FX0_x[i], FX1_x[i], FX2_x[i], FX3_x[i], FX4_x[i], rhs_x[i], dt)
    end

    __demand(__parallel)
    for i in pencil do
      get_yfluxes(Wprim_x[i], Wcnsr_x[i], FY0_y[i], FY1_y[i], FY2_y[i], FY3_y[i], FY4_y[i])
    end

    __demand(__parallel)
    for i in pencil do
      ddy(FY0_y[i], pLU_y[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddy(FY1_y[i], pLU_y[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddy(FY2_y[i], pLU_y[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddy(FY3_y[i], pLU_y[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddy(FY4_y[i], pLU_y[i])
    end

    __demand(__parallel)
    for i in pencil do
      add_yfluxes_to_rhs(FY0_y[i], FY1_y[i], FY2_y[i], FY3_y[i], FY4_y[i], rhs_y[i], dt)
    end

    __demand(__parallel)
    for i in pencil do
      get_zfluxes(Wprim_x[i], Wcnsr_x[i], FZ0_z[i], FZ1_z[i], FZ2_z[i], FZ3_z[i], FZ4_z[i])
    end

    __demand(__parallel)
    for i in pencil do
      ddz(FZ0_z[i], pLU_z[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddz(FZ1_z[i], pLU_z[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddz(FZ2_z[i], pLU_z[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddz(FZ3_z[i], pLU_z[i])
    end
    __demand(__parallel)
    for i in pencil do
      ddz(FZ4_z[i], pLU_z[i])
    end

    __demand(__parallel)
    for i in pencil do
      add_zfluxes_to_rhs(FZ0_z[i], FZ1_z[i], FZ2_z[i], FZ3_z[i], FZ4_z[i], rhs_z[i], dt)
    end

    __demand(__parallel)
    for i in pencil do
      update_conserved( Wcnsr_x[i], rhs_x[i] )
    end

    __demand(__parallel)
    for i in pencil do
      get_primitive( Wcnsr_x[i], Wprim_x[i] )
    end

    t += dt

    var tke : double = 0.0
    __demand(__parallel)
    for i in pencil do
      tke += get_TKE( Wprim_x[i] )
    end
    tke = tke / (N*N*N)  -- Average TKE

    wait_for(tke)
    var ts_end = c.legion_get_current_time_in_micros()

    c.printf("t = %12.5e\n", t)
    c.printf("    Normalized TKE = %8.5f\n", tke/tke0)
    c.printf("    CPU time = %8.5f seconds\n", (ts_end - ts_start)*1e-6)
  -- end -- while loop

end

regentlib.start(main)
