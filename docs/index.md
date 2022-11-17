```{include} ../README.md
```

# Site content

```{eval-rst}
.. toctree::
   :maxdepth: 3
   :caption: For Users
```

```{eval-rst}
.. autosummary::
    :toctree: api
    :caption: API
    :template: custom-module-template.rst
    :recursive:

    sertit
```

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :caption: For Contributors
   :hidden:

   history
   GitHub Repository <https://github.com/sertit/sertit-utils>
```

