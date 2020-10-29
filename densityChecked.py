#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:26:17 2020

@author: nathanl
"""
import math


def densityChecked(M) -> None:
    xlist = M[:,0]
    ylist = M[:,1]
    
    #EDIT AS NECESSARY
    #**********************************
    #radius parameter
    r0 = 5
    #minimum number of points allowable
    k = 5
    #**********************************

    #counter for points within r0
    count = 0

    #list of yes/no (1/0) for each point
    checklist = [0]*len(xlist)
    
    #checklist is now list of 1s and 0s with indices corresponding to xlist and ylist
    a=0;
    while a < len(xlist):
        b=0;
        while b < len(xlist):
            if a != b:
                distance = math.sqrt( (ylist[b] - ylist[a])*(ylist[b] - ylist[a])
                                     + (xlist[b] - xlist[a])*(xlist[b] - xlist[a]) )
            b+=1
            
            if distance <= r0:
                count += 1;
                
        if count >= k:
            checklist[a] = 1
        a+=1
        
    z=0;
    while z < len(xlist):
        if checklist[z] == 0:
            M.remove(M[z])
    
    