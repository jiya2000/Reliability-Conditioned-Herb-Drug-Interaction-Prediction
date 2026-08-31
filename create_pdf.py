from markdown_pdf import Section, MarkdownPdf

def generate_pdf():
    pdf = MarkdownPdf(toc_level=2)
    with open("Review_1_Report.md", "r") as f:
        md_text = f.read()
    
    # Add a section with the markdown text
    pdf.add_section(Section(md_text, toc=False))
    
    pdf.save("Review_1_Report.pdf")
    print("PDF generated successfully.")

if __name__ == "__main__":
    generate_pdf()
