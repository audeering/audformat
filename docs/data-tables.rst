Tables
======

A table links labels to media files.
It consists of one or more index columns
followed by an arbitrary number of label columns.
Labels can either refer to whole files or part of files.
An empty label means that no label has been assigned (yet).

There are two types of tables:

* **Filewise**: labels refer to whole files
* **Segmented**: labels refer to specific parts of files (segments)

Each type comes with a characteristic index:


Filewise
--------

==============  ====================================================
Index columns   Description
==============  ====================================================
file            Path to media file
==============  ====================================================

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Assign labels to the whole table
    values_dict = {'columnid': values_list}
    db['tableid'].set(values_dict, files=files_list)    
    # Assign labels to a column
    db['tableid']['columnid'].set(values_list, files=files_list)    
    # Access all labels as pandas.DataFrame
    db['tableid'].get()    
    # Access labels from a column as pandas.Series
    db['tableid']['columnid'].get()    
    # Access all labels as segments
    db['tableid'].get(
        files=files_list, 
        starts=starts_list,
        ends=ends_list)    
    # Access labels from a column as a segmented series
    db['tableid']['columnid'].get(
        files=files_list,
        starts=starts_list,
        ends=ends_list)    
    # Access labels from a column as segments
    db['tableid']['columnid'].get(
        files=files_list, 
        starts=starts_list,
        ends=ends_list)
    

Segmented
---------

==============  ====================================================
Index columns   Description
==============  ====================================================
file            Path to media file
start           Start time of the segment
                (relative to the beginning of the file)
end             End time of the segment
                (relative to the beginning of the file)
==============  ====================================================

audformat implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Assign labels to the whole table
    values_dict = {'columnid': values_list}
    db['tableid'].set(
        values_dict,
        files=files_list,
        starts=starts_list,
        ends=ends_list)
    # Assign labels to a column
    db['tableid']['columnid'].set(
        values_list,
        files=files_list,
        starts=starts_list,
        ends=ends_list)
    # Access all labels as pandas.DataFrame
    db['tableid'].get()
    # Access labels from a column as pandas.Series
    db['tableid']['columnid'].get()
