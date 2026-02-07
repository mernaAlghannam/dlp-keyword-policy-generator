using Microsoft.AspNetCore.Mvc;

namespace DlpKeywordPolicyGenerator.Web.Controllers;

public class HomeController : Controller
{
    public IActionResult Index()
    {
        return View();
    }
}
