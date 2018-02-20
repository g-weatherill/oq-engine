SHARE OpenQuake Computational Settings
======================================

============== ===================
checksum32     1,220,765,868      
date           2018-01-11T04:54:45
engine_version 2.9.0-git3c583c4   
============== ===================

num_sites = 1, num_imts = 3

Parameters
----------
=============================== ===========================================
calculation_mode                'classical'                                
number_of_logic_tree_samples    0                                          
maximum_distance                {'default': [(6, 100), (7, 150), (9, 200)]}
investigation_time              50.0                                       
ses_per_logic_tree_path         1                                          
truncation_level                3.0                                        
rupture_mesh_spacing            5.0                                        
complex_fault_mesh_spacing      5.0                                        
width_of_mfd_bin                0.2                                        
area_source_discretization      10.0                                       
ground_motion_correlation_model None                                       
random_seed                     23                                         
master_seed                     0                                          
=============================== ===========================================

Input files
-----------
======================= ==========================================================================
Name                    File                                                                      
======================= ==========================================================================
gsim_logic_tree         `complete_gmpe_logic_tree.xml <complete_gmpe_logic_tree.xml>`_            
job_ini                 `job.ini <job.ini>`_                                                      
source                  `simple_area_source_model.xml <simple_area_source_model.xml>`_            
source_model_logic_tree `simple_source_model_logic_tree.xml <simple_source_model_logic_tree.xml>`_
======================= ==========================================================================

Composite source model
----------------------
========= ====== ====================== ================
smlt_path weight gsim_logic_tree        num_realizations
========= ====== ====================== ================
b1        1.000  complex(0,4,1,0,4,2,5) 4/4             
========= ====== ====================== ================

Required parameters per tectonic region type
--------------------------------------------
====== ==================================================================================== ========== ========== ==============
grp_id gsims                                                                                distances  siteparams ruptparams    
====== ==================================================================================== ========== ========== ==============
4      AtkinsonBoore2003SSlab() LinLee2008SSlab() YoungsEtAl1997SSlab() ZhaoEtAl2006SSlab() rhypo rrup vs30       hypo_depth mag
====== ==================================================================================== ========== ========== ==============

Realizations per (TRT, GSIM)
----------------------------

::

  <RlzsAssoc(size=4, rlzs=4)
  4,AtkinsonBoore2003SSlab(): [0]
  4,LinLee2008SSlab(): [1]
  4,YoungsEtAl1997SSlab(): [2]
  4,ZhaoEtAl2006SSlab(): [3]>

Number of ruptures per tectonic region type
-------------------------------------------
============================ ====== ================= ============ ============
source_model                 grp_id trt               eff_ruptures tot_ruptures
============================ ====== ================= ============ ============
simple_area_source_model.xml 4      Subduction Inslab 7,770        93,219      
============================ ====== ================= ============ ============

Informational data
------------------
======================= ==================================================================================
count_ruptures.received tot 14.02 KB, max_per_task 3.61 KB                                                
count_ruptures.sent     sources 78.08 KB, param 4.56 KB, srcfilter 2.89 KB, gsims 1.26 KB, monitor 1.25 KB
hazard.input_weight     197634.70000000007                                                                
hazard.n_imts           3                                                                                 
hazard.n_levels         78                                                                                
hazard.n_realizations   1280                                                                              
hazard.n_sites          1                                                                                 
hazard.n_sources        18                                                                                
hazard.output_weight    78.0                                                                              
hostname                tstation.gem.lan                                                                  
require_epsilons        False                                                                             
======================= ==================================================================================

Slowest sources
---------------
========= ================== ============ ========= ========= =========
source_id source_class       num_ruptures calc_time num_sites num_split
========= ================== ============ ========= ========= =========
s46       AreaSource         7,770        0.082     1         370      
i20       ComplexFaultSource 9,241        0.0       1         0        
scr299    AreaSource         1,572        0.0       1         0        
scr301    AreaSource         17,268       0.0       1         0        
v4        AreaSource         168          0.0       1         0        
s34       AreaSource         12,327       0.0       1         0        
scr293    AreaSource         61,740       0.0       1         0        
i17       ComplexFaultSource 33,383       0.0       1         0        
s70       AreaSource         17,871       0.0       1         0        
s13       AreaSource         12,726       0.0       1         0        
s35       AreaSource         12,327       0.0       1         0        
s40       AreaSource         12,327       0.0       1         0        
sh14      AreaSource         41,952       0.0       1         0        
v1        AreaSource         42           0.0       1         0        
sh6       AreaSource         12,900       0.0       1         0        
s72       AreaSource         17,871       0.0       1         0        
sh13      AreaSource         41,952       0.0       1         0        
scr304    AreaSource         574          0.0       1         0        
========= ================== ============ ========= ========= =========

Computation times by source typology
------------------------------------
================== ========= ======
source_class       calc_time counts
================== ========= ======
AreaSource         0.082     16    
ComplexFaultSource 0.0       2     
================== ========= ======

Duplicated sources
------------------
There are no duplicated sources

Information about the tasks
---------------------------
================== ===== ====== ===== ===== =========
operation-duration mean  stddev min   max   num_tasks
count_ruptures     0.023 0.006  0.014 0.027 4        
================== ===== ====== ===== ===== =========

Slowest operations
------------------
============================== ========= ========= ======
operation                      time_sec  memory_mb counts
============================== ========= ========= ======
reading composite source model 9.091     0.0       1     
managing sources               0.157     0.0       1     
total count_ruptures           0.091     0.316     4     
store source_info              0.015     0.0       1     
aggregate curves               3.843E-04 0.0       4     
reading site collection        4.077E-05 0.0       1     
saving probability maps        2.599E-05 0.0       1     
============================== ========= ========= ======