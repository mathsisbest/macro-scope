{#
  Use the custom +schema name verbatim (e.g. 'staging', 'marts') instead of dbt's
  default '<target>_<custom>' concatenation. Keeps schema names clean and matches
  the Python/dashboard layers that read `marts.*`.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
