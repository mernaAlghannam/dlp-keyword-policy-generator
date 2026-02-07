using System.Text;
using DocumentFormat.OpenXml.Packaging;
using UglyToad.PdfPig;

namespace DlpKeywordPolicyGenerator.Web.Services;

public sealed class TextExtractionService
{
    public string ExtractText(IFormFile file)
    {
        var name = file.FileName.ToLowerInvariant();
        using var stream = file.OpenReadStream();
        using var memory = new MemoryStream();
        stream.CopyTo(memory);
        var data = memory.ToArray();

        if (name.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
        {
            return ReadPdfBytes(data).Trim();
        }

        if (name.EndsWith(".docx", StringComparison.OrdinalIgnoreCase))
        {
            return ReadDocxBytes(data).Trim();
        }

        return ReadTextBytes(data).Trim();
    }

    private static string ReadPdfBytes(byte[] data)
    {
        try
        {
            using var pdf = PdfDocument.Open(data);
            var sb = new StringBuilder();
            foreach (var page in pdf.GetPages())
            {
                sb.AppendLine(page.Text);
            }
            return sb.ToString();
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string ReadDocxBytes(byte[] data)
    {
        try
        {
            using var memory = new MemoryStream(data);
            using var doc = WordprocessingDocument.Open(memory, false);
            var body = doc.MainDocumentPart?.Document.Body;
            if (body == null)
            {
                return string.Empty;
            }

            var sb = new StringBuilder();
            foreach (var text in body.Descendants<DocumentFormat.OpenXml.Wordprocessing.Text>())
            {
                if (!string.IsNullOrWhiteSpace(text.Text))
                {
                    sb.AppendLine(text.Text);
                }
            }

            return sb.ToString();
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string ReadTextBytes(byte[] data)
    {
        var encodings = new[]
        {
            Encoding.UTF8,
            Encoding.Unicode,
            Encoding.GetEncoding("windows-1256"),
            Encoding.GetEncoding("windows-1252")
        };

        foreach (var encoding in encodings)
        {
            try
            {
                return encoding.GetString(data);
            }
            catch
            {
                continue;
            }
        }

        return Encoding.UTF8.GetString(data);
    }
}
