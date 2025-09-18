from gimkit.schemas import validate_wrapped_masked_io


def test_common_io():
    m_input = '<|M_INPUT|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/M_INPUT|>'
    m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
    validate_wrapped_masked_io(m_input, m_output)
