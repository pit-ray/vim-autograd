vim9script


export def ShapeToSize(shape: list<number>): number
  if len(shape) == 0
    return 0
  endif

  var size: number = 1
  for x in shape
    size *= x
  endfor
  return size
enddef


export def AsList(data: any): list<any>
  return type(data) != v:t_list ? [data] : data
enddef


export def CreateVector(
    size: number,
    init_val: float = 0.0): list<float>
  return repeat([init_val], size)
enddef


export def GetMatrixShape(array: any): list<number>
  var shape = []
  var sub_array = copy(array)
  while type(sub_array) == v:t_list && len(sub_array) > 0
    shape->add(len(sub_array))
    sub_array = sub_array[0]
  endwhile
  return shape
enddef


export def SqueezeRightShape(shape: list<number>): list<number>
  # e.g. [1, 5, 6, 1, 1, 1] -> [1, 5, 6]
  var dim = len(shape)
  var valid_size = dim
  for i in range(-1, -dim, -1)
    if shape[i] != 1
      break
    endif
    valid_size -= 1
  endfor

  if valid_size == 0
    return [1]
  endif

  return shape[: valid_size - 1]
enddef


export def SqueezeLeftShape(shape: list<number>): list<number>
  # e.g. [1, 1, 1, 6, 7, 1] -> [6, 7, 1]
  var dim = len(shape)
  var valid_size = dim
  for i in range(dim)
    if shape[i] != 1
      break
    endif
    valid_size -= 1
  endfor

  if valid_size == 0
    return [1]
  endif

  return shape[-valid_size :]
enddef
