# 🧪 API Reference

Welcome to the `climatrix` API reference. Below you'll find details on key modules, classes, and methods — with examples and usage tips to help you integrate it smoothly into your climate data workflows.

---

!!! abstract
    The main module `climatrix` provides tools to extend `xarray` datasets for climate subsetting, sampling, reconstruction. It is accessible via **accessor**.

---

The library contains a few public classes:

| Class name | Description |
| -----------| ----------- |
| [`Axis`](#climatrix.dataset.domain.Axis) | Enumerator class for spatio-temporal axes |
| [`BaseClimatrixDataset`](#climatrix.dataset.base.BaseClimatrixDataset) | Base class for managing `xarray` data |
| [`Domain`](#climatrix.dataset.domain.Domain) | Base class for domain-specific operations |
| [`SparseDomain`](#climatrix.dataset.domain.SparseDomain) | Subclass of `Domain` aim at managing sparse representations | 
| [`DenseDomain`](#climatrix.dataset.domain.DenseDomain) |  Subclass of `Domain` aim at managing dense representations | 


::: climatrix.dataset.domain.Axis
    handler: python
    options:    
      members:
        - get
      scoped_crossrefs: true
      show_root_heading: true
      show_source: false     


::: climatrix.dataset.base.BaseClimatrixDataset
    handler: python
    options:
      members:
        - domain
        - subset
        - to_signed_longitude
        - to_positive_longitude
        - time
        - itime
        - sample_uniform
        - sample_normal
        - reconstruct
        - plot
      scoped_crossrefs: true
      show_root_heading: true
      show_source: false


## 🌍 Domain 

::: climatrix.dataset.domain.Domain
    handler: python
    options:
      members:
        - from_lat_lon
        - latitude_name
        - longitude_name
        - time_name
        - point_name
        - latitude
        - longitude
        - time
        - point
        - get_size
        - is_dynamic
        - is_sparse
      scoped_crossrefs: true
      show_root_heading: true
      show_source: false      


::: climatrix.dataset.domain.SparseDomain
    handler: python
    options:    
      members:
        - to_xarray
      scoped_crossrefs: true
      show_root_heading: true
      show_source: false      

::: climatrix.dataset.domain.DenseDomain
    handler: python
    options:    
      members:
        - to_xarray    
      scoped_crossrefs: true
      show_root_heading: true
      show_source: false            

## 🌐 Reconstructors

::: climatrix.reconstruct.idw.IDWReconstructor
    handler: python
    options:   
      scoped_crossrefs: true 
      show_root_heading: true
      show_source: false    

::: climatrix.reconstruct.kriging.OrdinaryKrigingReconstructor
    handler: python
    options:    
      scoped_crossrefs: true
      show_root_heading: true
      show_source: false          

::: climatrix.reconstruct.siren.siren.SIRENReconstructor
    handler: python
    options:    
      scoped_crossrefs: true
      show_root_heading: true
      show_source: false         