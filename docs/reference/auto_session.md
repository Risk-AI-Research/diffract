# Python API

## Stateless analysis

```{eval-rst}
.. autofunction:: diffract.analyze

.. autofunction:: diffract.compare

.. autoclass:: diffract.AnalysisResult
   :members:

.. autoclass:: diffract.core.data.nn.extractors.report.ExtractionReport
   :members:

.. autoclass:: diffract.session.summaries.PairCoverage
   :members:
   :inherited-members:
```

## Field selectors

```{eval-rst}
.. autoclass:: diffract.FieldSelector
   :members: from_context, matches, is_contextual
```

## Session

```{eval-rst}
.. autoclass:: diffract.Session
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoexception:: diffract.session.SessionError
```

```{eval-rst}
.. autoexception:: diffract.session.ModelNotFoundError
```

```{eval-rst}
.. autoexception:: diffract.session.ModelAlreadyExistsError
```

```{eval-rst}
.. autoexception:: diffract.session.KernelNotFoundError
```

```{eval-rst}
.. autoexception:: diffract.session.ScopeValidationError
```

```{eval-rst}
.. autoexception:: diffract.session.InvalidIdentifierError
```

```{eval-rst}
.. autoexception:: diffract.session.IncompatibleStoreError
```

```{eval-rst}
.. autoexception:: diffract.session.UnrecognizedStoreError
```

## SVD workspace

```{eval-rst}
.. autofunction:: diffract.core.compute.extensions.svd_workspace.estimate_svd_workspace_bytes
```

## Session namespaces

The `Session` API surface is organized into namespaces, available as
`session.models`, `session.compute`, `session.results`, `session.viz`, and
`session.utils`.

```{eval-rst}
.. autoclass:: diffract.session.namespaces.ModelsNamespace
   :members:
```

```{eval-rst}
.. autoclass:: diffract.session.namespaces.ComputeNamespace
   :members:
```

```{eval-rst}
.. autoclass:: diffract.session.namespaces.ResultsNamespace
   :members:
```

```{eval-rst}
.. autoclass:: diffract.session.namespaces.VizNamespace
   :members:
```

```{eval-rst}
.. autoclass:: diffract.session.namespaces.UtilsNamespace
   :members:
```
