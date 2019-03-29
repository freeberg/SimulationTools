#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:44:40 2019

@author: annie
"""
from scipy import *
from pylab import *

import assimulo.problem as apr
from assimulo.explicit_ode import Explicit_ODE


class second_order(apr.Explicit_problem):
  
  def __init__(self, rhs, x0, xd0, t0):
    self.yd0 = xd0
    super().__init__(self, rhs, x0, t0)
    