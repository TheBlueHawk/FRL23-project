from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, online=False, chunks=1024,
      min_size=1, samples_per_insert=None, tolerance=1e4, seed=0):
    # Check if samples_per_insert is provided and select the appropriate limiter
    if samples_per_insert:
      limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      limiter = limiters.MinSize(min_size)
      
    # Check if capacity is not specified or min_size is less than or equal to capacity
    assert not capacity or min_size <= capacity
    
    # Call the constructor of the parent class (Generic) with the necessary arguments
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiter,
        directory=directory,
        online=online,
        chunks=chunks,
    )
