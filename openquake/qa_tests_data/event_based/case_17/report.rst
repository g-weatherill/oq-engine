Event Based Hazard QA Test, Case 17
===================================

+---------------+---------------------+
| checksum32    |3_446_125_708        |
+---------------+---------------------+
| date          |2021-05-02T09:25:43  |
+---------------+---------------------+
| engine_version|3.12.0-git9b6ffe086d |
+---------------+---------------------+

num_sites = 1, num_levels = 3, num_rlzs = 5

Parameters
----------
+--------------------------------+-------------------------------------------+
| calculation_mode               |'preclassical'                             |
+--------------------------------+-------------------------------------------+
| number_of_logic_tree_samples   |5                                          |
+--------------------------------+-------------------------------------------+
| maximum_distance               |{'default': [[1.0, 200.0], [10.0, 200.0]]} |
+--------------------------------+-------------------------------------------+
| investigation_time             |1.0                                        |
+--------------------------------+-------------------------------------------+
| ses_per_logic_tree_path        |3                                          |
+--------------------------------+-------------------------------------------+
| truncation_level               |2.0                                        |
+--------------------------------+-------------------------------------------+
| rupture_mesh_spacing           |2.0                                        |
+--------------------------------+-------------------------------------------+
| complex_fault_mesh_spacing     |2.0                                        |
+--------------------------------+-------------------------------------------+
| width_of_mfd_bin               |1.0                                        |
+--------------------------------+-------------------------------------------+
| area_source_discretization     |20.0                                       |
+--------------------------------+-------------------------------------------+
| pointsource_distance           |None                                       |
+--------------------------------+-------------------------------------------+
| ground_motion_correlation_model|None                                       |
+--------------------------------+-------------------------------------------+
| minimum_intensity              |{}                                         |
+--------------------------------+-------------------------------------------+
| random_seed                    |106                                        |
+--------------------------------+-------------------------------------------+
| master_seed                    |123456789                                  |
+--------------------------------+-------------------------------------------+
| ses_seed                       |106                                        |
+--------------------------------+-------------------------------------------+

Input files
-----------
+------------------------+-------------------------------------------------------------+
| Name                   |File                                                         |
+------------------------+-------------------------------------------------------------+
| gsim_logic_tree        |`gsim_logic_tree.xml <gsim_logic_tree.xml>`_                 |
+------------------------+-------------------------------------------------------------+
| job_ini                |`job.ini <job.ini>`_                                         |
+------------------------+-------------------------------------------------------------+
| source_model_logic_tree|`source_model_logic_tree.xml <source_model_logic_tree.xml>`_ |
+------------------------+-------------------------------------------------------------+

Composite source model
----------------------
+-------+------------------+-------------+
| grp_id|gsim              |rlzs         |
+-------+------------------+-------------+
| 0     |'[SadighEtAl1997]'|[0]          |
+-------+------------------+-------------+
| 1     |'[SadighEtAl1997]'|[1, 2, 3, 4] |
+-------+------------------+-------------+

Required parameters per tectonic region type
--------------------------------------------
+--------+------------------+---------+----------+-----------+
| trt_smr|gsims             |distances|siteparams|ruptparams |
+--------+------------------+---------+----------+-----------+
| 0      |'[SadighEtAl1997]'|rrup     |vs30      |mag rake   |
+--------+------------------+---------+----------+-----------+
| 1      |'[SadighEtAl1997]'|rrup     |vs30      |mag rake   |
+--------+------------------+---------+----------+-----------+

Slowest sources
---------------
+----------+----+---------+---------+-------------+
| source_id|code|calc_time|num_sites|eff_ruptures |
+----------+----+---------+---------+-------------+
| 1        |P   |2.408E-04|1        |39           |
+----------+----+---------+---------+-------------+
| 2        |P   |1.619E-04|1        |7            |
+----------+----+---------+---------+-------------+

Computation times by source typology
------------------------------------
+-----+---------+---------+-------------+
| code|calc_time|num_sites|eff_ruptures |
+-----+---------+---------+-------------+
| P   |4.027E-04|2        |46           |
+-----+---------+---------+-------------+

Information about the tasks
---------------------------
+-------------------+------+-------+------+-------+--------+
| operation-duration|counts|mean   |stddev|min    |max     |
+-------------------+------+-------+------+-------+--------+
| preclassical      |2     |0.00115|3%    |0.00111|0.00119 |
+-------------------+------+-------+------+-------+--------+
| read_source_model |2     |0.00152|5%    |0.00144|0.00159 |
+-------------------+------+-------+------+-------+--------+

Data transfer
-------------
+------------------+---------------------------------------------+---------+
| task             |sent                                         |received |
+------------------+---------------------------------------------+---------+
| read_source_model|converter=622 B fname=202 B                  |3.18 KB  |
+------------------+---------------------------------------------+---------+
| preclassical     |srcs=2.39 KB srcfilter=2.33 KB params=1.57 KB|3.12 KB  |
+------------------+---------------------------------------------+---------+

Slowest operations
------------------
+-------------------------+---------+---------+-------+
| calc_3409, maxmem=0.6 GB|time_sec |memory_mb|counts |
+-------------------------+---------+---------+-------+
| composite source model  |1.22525  |0.0      |1      |
+-------------------------+---------+---------+-------+
| total read_source_model |0.00303  |0.33203  |2      |
+-------------------------+---------+---------+-------+
| total preclassical      |0.00229  |0.0      |2      |
+-------------------------+---------+---------+-------+
| splitting sources       |8.168E-04|0.0      |2      |
+-------------------------+---------+---------+-------+
| weighting sources       |6.590E-04|0.0      |2      |
+-------------------------+---------+---------+-------+