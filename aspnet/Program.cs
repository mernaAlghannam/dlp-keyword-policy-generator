using DlpKeywordPolicyGenerator.Web.Services;
using DlpKeywordPolicyGenerator.Web.Services.Analysis;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllersWithViews();
Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);

builder.Services.AddSingleton<TextExtractionService>();
builder.Services.AddSingleton<SessionStore>();
builder.Services.AddSingleton<TopicInferenceService>();
builder.Services.AddSingleton<AnalysisService>();
builder.Services.AddSingleton<PolicyService>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
}

app.UseStaticFiles();

app.UseRouting();

app.MapControllers();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");

app.Run();
