from jinja2 import Environment, PackageLoader, StrictUndefined, Template

# Templates are plain text (agent prompts), never rendered as HTML — no autoescape needed.
jinja_env = Environment(
    loader=PackageLoader("vero", "templates"), undefined=StrictUndefined
)


def get_stored_jinja_template(template_name: str) -> Template:
    """Loads a stored Template from the jinja/ directory of the package."""

    template_name = template_name.removesuffix(".j2")
    template_filename = f"{template_name}.j2"
    return jinja_env.get_template(template_filename)
