from langchain.output_parsers.regex import RegexParser
from sociaty_newsletter_generator.newsletter_generator import (
    create_newsletter_output_parser,
)


def test_newsletter_output_parser():
    parser = create_newsletter_output_parser()

    newsletter = "Lorem\nipsum"

    output = f"Here is the newsletter : \n<newsletter>\n{newsletter}\n</newsletter>\nHope it fits your needs !"

    result = parser.invoke(output)

    assert "newsletter" in result
    assert result["newsletter"].strip() == newsletter.strip()


def test_newsletter_output_parser_with_newsletter_template_as_tag():
    parser = create_newsletter_output_parser()

    newsletter = "Lorem\nipsum"

    output = f"Here is the newsletter : \n<newsletter_template>\n{newsletter}\n</newsletter_template>\nHope it fits your needs !"

    result = parser.invoke(output)

    assert "newsletter" in result
    assert result["newsletter"].strip() == newsletter.strip()
