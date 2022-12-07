{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}

    {% if methods|length > 1 and attributes %}
    .. rubric:: {{ _('Methods, Properties, Attributes') }}

    .. autosummary::
        :toctree:
        :nosignatures:
    {% for item in (all_methods + attributes)|sort(attribute=0) %}
        {%- if not item.startswith('_') or item in ['__contains__', '__getitem__', '__setitem__'] %}
        {{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    {% elif not attributes %}
    .. rubric:: {{ _('Methods, Properties') }}

    .. autosummary::
        :toctree:
        :nosignatures:
    {% for item in all_methods %}
        {%- if not item.startswith('_') or item in ['__contains__', '__getitem__', '__setitem__'] %}
        {{ name }}.{{ item }}
        {% endif %}
    {%- endfor %}

    {% else %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :toctree:
        :nosignatures:
    {% for item in attributes %}
        {{ name }}.{{ item }}
    {%- endfor %}

    {% endif %}
    {% endblock %}
