import scrapy


class HistoryenSpider(scrapy.Spider):
    name = "historyEN"
    allowed_domains = ["en.wikipedia.org"]
    start_urls = ["https://en.wikipedia.org/wiki/Category:Treaties_of_Iran", "https://en.wikipedia.org/wiki/Category:Treaties_of_the_Achaemenid_Empire", "https://en.wikipedia.org/wiki/Category:Treaties_of_the_Parthian_Empire", "https://en.wikipedia.org/wiki/Category:Treaties_of_the_Sasanian_Empire" , "https://en.wikipedia.org/wiki/Category:Treaties_of_the_Safavid_dynasty", "https://en.wikipedia.org/wiki/Category:Treaties_of_Afsharid_Iran" , "https://en.wikipedia.org/wiki/Category:Treaties_of_the_Qajar_dynasty" , "https://en.wikipedia.org/wiki/Category:Treaties_of_Pahlavi_Iran" , "https://en.wikipedia.org/wiki/Category:Peace_treaties_of_Iran", "https://en.wikipedia.org/wiki/Category:Single_Convention_on_Narcotic_Drugs" , "https://en.wikipedia.org/wiki/Category:Iran%E2%80%93Ottoman_Empire_treaties" , "https://en.wikipedia.org/wiki/Category:Iran%E2%80%93Russia_treaties"]
    base_url = "https://en.wikipedia.org"
    
    def parse(self, response):
        for links in response.css('div.mw-category-group ul li'):
            yield{
                'names' : links.css('a::text').get().strip(),
                'link' : self.base_url + links.css('a::attr(href)').get(),
            }
