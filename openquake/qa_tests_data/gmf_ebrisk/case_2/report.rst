Event Based Risk from GMF
=========================

============== ===================
checksum32     3,574,592,625      
date           2018-01-11T04:30:28
engine_version 2.9.0-git3c583c4   
============== ===================

num_sites = 3, num_imts = 2

Parameters
----------
=============================== ============
calculation_mode                'gmf_ebrisk'
number_of_logic_tree_samples    0           
maximum_distance                None        
investigation_time              1.0         
ses_per_logic_tree_path         1           
truncation_level                None        
rupture_mesh_spacing            None        
complex_fault_mesh_spacing      None        
width_of_mfd_bin                None        
area_source_discretization      None        
ground_motion_correlation_model None        
random_seed                     42          
master_seed                     0           
avg_losses                      True        
=============================== ============

Input files
-----------
======================== ================================================
Name                     File                                            
======================== ================================================
exposure                 `exposure_model_2.xml <exposure_model_2.xml>`_  
gmfs                     `gmfs_3_2IM.csv <gmfs_3_2IM.csv>`_              
job_ini                  `job.ini <job.ini>`_                            
sites                    `sitemesh.csv <sitemesh.csv>`_                  
structural_vulnerability `vulnerability_2IM.xml <vulnerability_2IM.xml>`_
======================== ================================================

Composite source model
----------------------
========= ====== =============== ================
smlt_path weight gsim_logic_tree num_realizations
========= ====== =============== ================
b_1       1.000  trivial(1)      1/1             
========= ====== =============== ================

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=1, rlzs=1)
  0,FromFile: [0]>

Exposure model
--------------
=============== ========
#assets         3       
#taxonomies     2       
deductibile     absolute
insurance_limit absolute
=============== ========

======== ===== ====== === === ========= ==========
taxonomy mean  stddev min max num_sites num_assets
RC       1.000 NaN    1   1   1         1         
RM       1.000 0.0    1   1   2         2         
*ALL*    1.000 0.0    1   1   3         3         
======== ===== ====== === === ========= ==========

Slowest operations
------------------
======================= ========= ========= ======
operation               time_sec  memory_mb counts
======================= ========= ========= ======
reading exposure        0.006     0.0       1     
building riskinputs     0.005     0.0       1     
assoc_assets_sites      0.005     0.0       1     
reading site collection 1.805E-04 0.0       1     
======================= ========= ========= ======