var profilingData = {
  items: [],
  visibility: [],
};

var maxZoom = 10000.0
var margin_left = 300;
var margin_right = 50;
var margin_bottom = 50;
var width = $(window).width() - margin_left - margin_right;
var height = $(window).height() - margin_bottom;
var start = 0;
var end = 4738871 * 1.01;
var max_level = 4657;
var scale = width / (end - start);
var zoom = 1.0;
var thickness = Math.max(height / max_level, 20);
var baseThickness = height / max_level;
height = max_level * thickness;
var min_feature_width = 3; // width in pixels of the smallest thing we want people to be able to see
var min_gap_width = 1; // width in pixels of the smallest gap we want to show
var filteredData;;

var mouseX = 0;
var rangeZoom = true;

var helpMessage = [
  "Zoom-in (x-axis)   : Ctrl + / 4",
  "Zoom-out (x-axis)  : Ctrl - / 3",
  "Reset zoom (x-axis): Ctrl 0 / 0",
  "Zoom-in (y-axis)   : Ctrl-Alt + / 2",
  "Zoom-out (y-axis)  : Ctrl-Alt - / 1",
  "Reset zoom (y-axis): Ctrl-Alt 0 / `",
  "Range Zoom-in      : drag-select",
  "Measure duration   : Alt + drag-select",
  "Search             : s / S",
  "Search History     : h / H",
  "Previous Search    : p / P",
  "Next Search        : n / N",
  "Clear Search       : c / C",
  "Toggle Search      : t / T",
];

function makeTimelineTransparent() {
  var timelineSvg = d3.select("#timeline").select("svg");
  timelineSvg.select("g#timeline").style("opacity", "0.1");
  $("#timeline").css("overflow-x", "hidden");
  timelineSvg.select("g#lines").style("opacity", "0.1");
  timelineSvg.select("g.locator").style("opacity", "0.1");
}

function makeTimelineOpaque() {
  var timelineSvg = d3.select("#timeline").select("svg");
  timelineSvg.select("g#timeline").style("opacity", "1.0");
  $("#timeline").css("overflow-x", "scroll");
  timelineSvg.select("g#lines").style("opacity", "1.0");
  timelineSvg.select("g.locator").style("opacity", "1.0");
}

function mouseMoveHandlerWhenDown() {
  var timelineSvg = d3.select("#timeline").select("svg");
  var p = d3.mouse(this);
  var select_block = timelineSvg.select("rect.select-block");
  var select_text = timelineSvg.select("text.select-block");
  var newWidth = Math.abs(p[0] - mouseX);
  select_block.attr("width", newWidth);
  if (p[0] >= mouseX) {
    select_block.attr("x", mouseX);
    select_text.attr("x", mouseX + (p[0] - mouseX) / 2);
  }
  else {
    select_block.attr("x", p[0]);
    select_text.attr("x", p[0] + (mouseX - p[0]) / 2);
  }
  var time = newWidth / zoom / scale;
  var unit = "us";
  if (time > 1000) {
    time = Math.floor(time / 1000);
    unit = "ms";
  }
  else {
    time = Math.floor(time);
  }
  select_text.text(time + " " + unit);
}

function mouseMoveHandlerWhenUp() {
  var p = d3.mouse(this);
  var x = parseFloat(p[0]);
  var scrollLeft = $("#timeline").scrollLeft();
  var paneWidth = $("#timeline").width();
  var currentTime = x / zoom / scale;
  var unit = "us";

  if (paneWidth / scale >= 100000) {
    currentTime = Math.floor(currentTime / 1000);
    unit = "ms";
  }
  else {
    currentTime = Math.floor(currentTime);
  }

  var timelineSvg = d3.select("#timeline").select("svg");
  timelineSvg.select("g.locator").remove();
  var locator = timelineSvg.append("g").attr("class", "locator");
  locator.append("line")
    .attr({
      x1: p[0],
      y1: 0,
      x2: p[0],
      y2: p[1] - thickness / 2,
      class: "locator",
    });
  locator.append("line")
    .attr({
      x1: p[0],
      y1: p[1] + thickness / 2,
      x2: p[0],
      y2: height,
      class: "locator",
    });
  var locatorText = locator.append("text");
  var text = currentTime + " " + unit;
  locatorText.attr("class", "locator").text(text)
  if ((x - scrollLeft) < paneWidth - 100) {
    locatorText.attr({x: x + 2, y: $(window).scrollTop() + 10});
    locatorText.attr("anchor", "start");
  }
  else {
    locatorText.attr({x: x - 2 - text.length * 7, y: $(window).scrollTop() + 10});
    locatorText.attr("anchor", "end");
  }
}

function mouseDownHandler() {
  var timelineSvg = d3.select("#timeline").select("svg");
  timelineSvg.select("g.locator").remove();
  var p = d3.mouse(this);
  timelineSvg.append("rect")
    .attr({
      x : p[0],
      y : 0,
      class : "select-block",
      width : 0,
      height : height
    });
  timelineSvg.append("text")
    .attr({
      x : p[0],
      y : p[1],
      class : "select-block",
      anchor : "middle",
      "text-anchor" : "middle",
    }).text("0 us");
  mouseX = p[0];
  timelineSvg.on("mousemove", null);
  timelineSvg.on("mousemove", mouseMoveHandlerWhenDown);
  $(document).off("keydown");
}

function mouseUpHandler() {
  var p = d3.mouse(this);
  var timelineSvg = d3.select("#timeline").select("svg");
  var select_block = timelineSvg.select("rect.select-block");
  var select_text = timelineSvg.select("text.select-block");
  var prevZoom = zoom;
  var selectWidth = parseInt(select_block.attr("width"));
  var svgWidth = timelineSvg.attr("width");
  if (rangeZoom && selectWidth > 10) {
    var x = select_block.attr("x");
    showLoaderIcon();
    adjustZoom(Math.min(svgWidth / selectWidth, maxZoom), false);
    $("#timeline").scrollLeft(x / prevZoom * zoom);
    hideLoaderIcon();
  }
  timelineSvg.selectAll("rect.select-block").remove();
  timelineSvg.selectAll("text.select-block").remove();
  mouseX = 0;
  timelineSvg.on("mousemove", null);
  timelineSvg.on("mousemove", mouseMoveHandlerWhenUp);
  $(document).on("keydown", defaultKeydown);
}

function turnOffMouseHandlers() {
  var timelineSvg = d3.select("#timeline").select("svg");
  timelineSvg.on("mousedown", null);
  timelineSvg.on("mouseup", null);
  timelineSvg.on("mousemove", null);
}

function turnOnMouseHandlers() {
  var timelineSvg = d3.select("#timeline").select("svg");
  timelineSvg.on("mousedown", mouseDownHandler);
  timelineSvg.on("mouseup", mouseUpHandler);
  timelineSvg.on("mousemove", mouseMoveHandlerWhenUp);
}

function drawLoaderIcon() {
  var loaderSvg = d3.select("#loader-icon").select("svg");
  var loaderGroup = loaderSvg.append("g")
    .attr({
        id: "loader-icon",
    });
  loaderGroup.append("path")
    .attr({
      opacity: 0.2,
      fill: "#000",
      d: "M20.201,5.169c-8.254,0-14.946,6.692-14.946,14.946c0,8.255,6.692,14.946,14.946,14.946s14.946-6.691,14.946-14.946C35.146,11.861,28.455,5.169,20.201,5.169z M20.201,31.749c-6.425,0-11.634-5.208-11.634-11.634c0-6.425,5.209-11.634,11.634-11.634c6.425,0,11.633,5.209,11.633,11.634C31.834,26.541,26.626,31.749,20.201,31.749z"
    });
  var path = loaderGroup.append("path")
    .attr({
      fill: "#000",
      d: "M26.013,10.047l1.654-2.866c-2.198-1.272-4.743-2.012-7.466-2.012h0v3.312h0C22.32,8.481,24.301,9.057,26.013,10.047z"
    });
  path.append("animateTransform")
    .attr({
      attributeType: "xml",
      attributeName: "transform",
      type: "rotate",
      from: "0 20 20",
      to: "360 20 20",
      dur: "0.5s",
      repeatCount: "indefinite"
    });
}
function showLoaderIcon() {
  loaderSvg.select("g").attr("visibility", "visible");
}
function hideLoaderIcon() {
  loaderSvg.select("g").attr("visibility", "hidden");
}

function getMouseOver(zoom) {
  var paneWidth = $("#timeline").width();
  var left = paneWidth / 3;
  var right = paneWidth * 2 / 3;
  return function(d, i) {
    var p = d3.mouse(this);
    var x = parseFloat(p[0]);
    var relativeX = (x - $("#timeline").scrollLeft())
    var anchor = relativeX < left ? "start" :
                 relativeX < right ? "middle" : "end";
    var descView = timelineSvg.append("g").attr("id", "desc");

    var total = d.end - d.start;
    var initiation = "";
    if ((d.initiation != undefined) && d.initiation != "") {
      initiation = " initiated by " + operations[d.initiation];
    } 
    var title = d.title + initiation + ", total=" + total + "us, start=" + 
                d.start + "us, end=" + d.end+ "us";

    descView.append("text")
      .attr("x", x)
      .attr("y", d.level * thickness - 5)
      .attr("text-anchor", anchor)
      .attr("class", "desc")
      .text(unescape(escape(title)));
  };
}

var searchEnabled = false;
var sizeHistory = 10;
var currentPos;
var nextPos;
var searchRegex = null;

function drawTimeline(targetSvg, data, zoom, scale, thickness) {
  var timeline = targetSvg.selectAll("rect")
    .data(data, function(d) { return d.id; });
  var mouseOver = getMouseOver(zoom);

  timeline.enter().append("rect");

  timeline
    .attr("id", function(d) { return "block-" + d.id; })
    .attr("x", function(d) { return d.start * scale * zoom; })
    .attr("y", function(d) { return d.level * thickness; })
    .style("fill", function(d) {
      if (!searchEnabled ||
          searchRegex[currentPos].exec(d.title) == null)
        return d.color;
      else return "#ff0000";
    })
    .attr("width", function(d) {
      if (searchEnabled && searchRegex[currentPos].exec(d.title) != null) {
        return Math.max(min_feature_width, (d.end - d.start) * scale * zoom);
      } else {
        //console.log("d.end:", d.end, "d.start:", d.start);
        return Math.max(min_feature_width, (d.end - d.start) * scale * zoom);
        //return (d.end - d.start) * scale * zoom;
      }
    })
    .attr("height", thickness)
    .style("opacity", function(d) {
      if (!searchEnabled ||
          searchRegex[currentPos].exec(d.title) != null)
        return d.opacity;
      else return 0.05;
    })
    .on("mouseout", function(d, i) { timelineSvg.selectAll("#desc").remove(); });
  timeline.on("mouseover", mouseOver);

  timeline.exit().remove();
}

function drawProcessors(data) {
  var svg = d3.select("#processors").append("svg")
    .attr("width", margin_left)
    .attr("height", height);

  var names = svg.selectAll(".processors")
    .data(data)
    .enter().append("text");

  names.attr("text-anchor", "start")
    .attr("class", "processor")
    .attr("x", 0)
    .attr("y", function(d) { return d.level * thickness + thickness; });

  names.each(function(d) {
    var text = d3.select(this);
    var tokens = d.processor.split(" to ");
    if (tokens.length == 1)
      text.append("tspan").text(d.processor);
    else {
      var source = tokens[0];
      var target = tokens[1].replace(" Channel", "");
      text.append("tspan").text(source)
        .attr({x : 0, dy : -10});
      text.append("tspan").text("==> " + target)
        .attr({x : 0, dy : 10});
    }
  });

  var lines = timelineSvg
    .append("g")
    .attr("id", "lines");

  lines.selectAll(".lines")
    .data(data)
    .enter().append("line")
    .attr("x1", 0)
    .attr("y1", function(d) { return d.level * thickness + thickness; })
    .attr("x2", zoom * width)
    .attr("y2", function(d) { return d.level * thickness + thickness; })
    .style("stroke", "#000000")
    .style("stroke-width", "1px");
}

function drawHelpBox() {
  var paneWidth = $("#timeline").width();
  var paneHeight = $("#timeline").height();
  var helpBoxGroup = timelineSvg.append("g").attr("class", "help-box");
  var helpBoxWidth = Math.min(450, paneWidth - 100);
  var helpTextOffset = 20;
  var helpBoxHeight = Math.min(helpMessage.length * helpTextOffset + 100,
                               paneHeight - 100);

  var timelineWidth = timelineSvg.select("g#timeline").attr("width");
  var timelineHeight = timelineSvg.select("g#timeline").attr("height");
  var scrollLeft = $("#timeline").scrollLeft();
  var scrollTop = $(window).scrollTop();

  var boxStartX = scrollLeft + (paneWidth - helpBoxWidth) / 2;
  var boxStartY = scrollTop + (paneHeight - helpBoxHeight) / 2 / (thickness / baseThickness);

  helpBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: helpBoxWidth,
        height: helpBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var helpText = helpBoxGroup.append("text")
    .attr("class", "help-box")
    .style("width", helpBoxWidth);
  var helpTitle = "Keyboard Shortcuts";
  helpText.append("tspan")
    .attr({ x: boxStartX + helpBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text(helpTitle);
  var off = 15;
  for (var i = 0; i < helpMessage.length; ++i) {
    helpText.append("tspan")
      .style("font-size", "12pt")
      .attr("text-anchor", "start")
      .attr({ x: boxStartX + 30, dy: off + helpTextOffset})
      .text(helpMessage[i]);
    off = 0;
  }
}

function drawSearchBox() {
  var paneWidth = $("#timeline").width();
  var paneHeight = $("#timeline").height();
  var searchBoxGroup = timelineSvg.append("g").attr("class", "search-box");
  var searchBoxWidth = Math.min(450, paneWidth - 100);
  var searchBoxHeight = Math.min(250, paneHeight - 100);

  var timelineWidth = timelineSvg.select("g#timeline").attr("width");
  var timelineHeight = timelineSvg.select("g#timeline").attr("height");
  var scrollLeft = $("#timeline").scrollLeft();
  var scrollTop = $(window).scrollTop();

  var boxStartX = scrollLeft + (paneWidth - searchBoxWidth) / 2;
  var boxStartY = scrollTop + (paneHeight - searchBoxHeight) / 2 / (thickness / baseThickness);

  searchBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: searchBoxWidth,
        height: searchBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var searchText = searchBoxGroup.append("text")
    .attr("class", "search-box")
    .style("width", searchBoxWidth);
  searchText.append("tspan")
    .attr({ x: boxStartX + searchBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text("Search");
  var searchInputWidth = searchBoxWidth - 40;
  var searchInputHeight = 50;
  searchBoxGroup.append("foreignObject")
    .attr({ x: boxStartX + 20, y: boxStartY + 150,
            width: searchInputWidth, height: searchInputHeight})
    .attr("text-anchor", "middle")
    .html("<input type='text' class='search-box' style='height: "
        + searchInputHeight + "px; width: "
        + searchInputWidth + "px; font-size: 20pt; font-family: Consolas, monospace;'></input>");
  $("input.search-box").focus();
}

function drawSearchHistoryBox() {
  var paneWidth = $("#timeline").width();
  var paneHeight = $("#timeline").height();
  var historyBoxGroup = timelineSvg.append("g").attr("class", "history-box");
  var historyBoxWidth = Math.min(450, paneWidth - 100);
  var historyBoxHeight = Math.min(350, paneHeight - 100);

  var timelineWidth = timelineSvg.select("g#timeline").attr("width");
  var timelineHeight = timelineSvg.select("g#timeline").attr("height");
  var scrollLeft = $("#timeline").scrollLeft();
  var scrollTop = $(window).scrollTop();

  var boxStartX = scrollLeft + (paneWidth - historyBoxWidth) / 2;
  var boxStartY = scrollTop + (paneHeight - historyBoxHeight) / 2 / (thickness / baseThickness);

  historyBoxGroup.append("rect")
    .attr({
        rx: 30,
        ry: 30,
        x: boxStartX,
        y: boxStartY,
        width: historyBoxWidth,
        height: historyBoxHeight,
        style: "fill: #222; opacity: 0.8;"
    });
  var historyText = historyBoxGroup.append("text")
    .attr("class", "history-box")
    .style("width", historyBoxWidth);
  var historyTitle = "Search History";
  historyText.append("tspan")
    .attr({ x: boxStartX + historyBoxWidth / 2, y: boxStartY + 50})
    .attr("text-anchor", "middle")
    .style("font-size", "20pt")
    .text(historyTitle);
  if (searchRegex != null) {
    var off = 15;
    var id = 1;
    for (var i = 0; i < sizeHistory; ++i) {
      var pos = (nextPos + i) % sizeHistory;
      var regex = searchRegex[pos];
      if (regex != null) {
        if (pos == currentPos) prefix = ">>> ";
        else prefix = id + " : ";
        historyText.append("tspan")
          .attr("text-anchor", "start")
          .attr({ x: boxStartX + 30, dy: off + 25})
          .text(prefix + regex.source);
        off = 0;
        id++;
      }
    }
  }
}

function updateURL(zoom, scale) {
  var windowStart = $("#timeline").scrollLeft();
  var windowEnd = windowStart + $("#timeline").width();
  var start_time = windowStart / zoom / scale;
  var end_time = windowEnd / zoom / scale;
  var url = window.location.href.split('?')[0];
  url += "?start=" + start_time;
  url += "&end=" + end_time;
  if (searchEnabled)
    url += "&search=" + searchRegex[currentPos].source;
  window.history.replaceState("", "", url);
}

function filterOnlyVisibleData(profilingData, zoom, scale) {
  updateURL(zoom, scale);
  var windowStart = $("#timeline").scrollLeft();
  var windowEnd = windowStart + $("#timeline").width();
  var filteredData = Array();
  var items = profilingData.items;
  var f = scale * zoom;
  var min_feature_time = min_feature_width / f;
  var min_gap_time = min_gap_width / f;
  for (var level in items) {
    // gap merging below assumes intervals are sorted - do that first
    items[level].sort(function(a,b) { return a.start - b.start; });
    for (var i = 0; i < items[level].length; ++i) {
      var d = items[level][i];
      var start = d.start;
      var end = d.end;
      if ((start * f) > windowEnd) continue;
      if ((end * f) < windowStart) continue;
      // is this block too narrow?
      if ((end - start) < min_feature_time) {
        // see how many more after this are also too narrow and too close to us
        var count = 1;
        // don't do this if we're the subject of the current search and don't merge with
        // something that is
        if (!searchEnabled || searchRegex[currentPos].exec(d.title) == null) {
          while (((i + count) < items[level].length) && 
                 ((items[level][i + count].start - end) < min_gap_time) &&
                 ((items[level][i + count].end - items[level][i + count].start) < min_feature_time) &&
                 (!searchEnabled || searchRegex[currentPos].exec(items[level][i + count].title) == null)) {
            end = items[level][i + count].end;
            count++;
          }
        }
        // are we still too narrow?  if so, bloat, but make sure we don't overlap something later
        if ((end - start) < min_feature_time) {
          end = start + min_feature_time;                   
          if (((i + count) < items[level].length) && (items[level][i + count].start < end))
            end = items[level][i + count].start;
        }
        if (count > 1) {
          filteredData.push({
            id: d.id,
            level: d.level,
            start: d.start,
            end: end,
            color: "#808080",
            title: count + " merged tasks"
          });
          i += (count - 1);
        } else {
          filteredData.push({
            id: d.id,
            level: d.level,
            start: d.start,
            end: d.end,
            color: d.color,
            initiation: d.initiation,
            title: d.title + " (expanded for visibility)"
          });
        }
      } else {
        filteredData.push(d);
      }
    }
  }
  return filteredData;
}

function adjustZoom(newZoom, scroll) {
  var prevZoom = zoom;
  zoom = Math.round(newZoom * 10) / 10;
  var svg = d3.select("#timeline").select("svg");

  svg.attr("width", zoom * width)
     .attr("height", height);

  var timelineGroup = svg.select("g#timeline");
  timelineGroup.selectAll("rect").remove();

  filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
  drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
  //svg.select("g#timeline")
  //  .attr("transform", "scale(" + zoom + ", 1.0)");

  svg.select("g#lines").selectAll("line")
    .attr("x2", zoom * width);
  svg.selectAll("#desc").remove();
  svg.selectAll("g.locator").remove();

  if (scroll) {
    var paneWidth = $("#timeline").width();
    var pos = ($("#timeline").scrollLeft() + paneWidth / 2) / prevZoom;
    // this will trigger a scroll event which in turn redraws the timeline
    $("#timeline").scrollLeft(pos * zoom - width / 2);
  }
}

function adjustThickness(newThickness) {
  thickness = newThickness;
  height = thickness * max_level;
  d3.select("#processors").select("svg").remove();
  var svg = d3.select("#timeline").select("svg");
  var timelineGroup = svg.select("g#timeline");
  timelineGroup.selectAll("rect").remove();
  var lines = timelineSvg.select("g#lines");
  lines.remove();

  svg.attr("width", zoom * width)
     .attr("height", height);
  //var filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
  drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
  drawProcessors(profilingData.processors);
  svg.selectAll("#desc").remove();
  svg.selectAll("g.locator").remove();
}

function suppressdefault(e) {
  if (e.preventDefault) e.preventDefault();
  if (e.stopPropagation) e.stopPropagation();
}

function setKeyHandler(handler) {
  $(document).off("keydown");
  $(document).on("keydown", handler);
}

function makeModalKeyHandler(keys, callback) {
  return function(e) {
    if (!e) e = event;
    var code = e.keyCode || e.charCode;
    if (!(e.ctrlKey || e.metaKey || e.altKey)) {
      for (var i = 0; i < keys.length; ++i) {
        if (code == keys[i]) {
          callback(code);
          return false;
        }
      }
    }
    return true;
  }
}

var Command = { none : 0, help : 1,
                zox : 2, zix : 3, zrx : 4,
                zoy : 5, ziy : 6, zry : 7,
                search : 8, clear_search : 9,
                toggle_search : 10, search_history : 11,
                previous_search : 12, next_search : 13
              };

function defaultKeydown(e) {
  if (!e) e = event;

  var code = e.keyCode || e.charCode;
  var commandType = Command.none;
  var multiFnKeys = e.metaKey && e.ctrlKey || e.altKey && e.ctrlKey;

  if (!(e.ctrlKey || e.metaKey || e.altKey)) {
    if (code == 191) commandType = Command.help;
    else if (code == 49) commandType = Command.zoy;
    else if (code == 50) commandType = Command.ziy;
    else if (code == 51) commandType = Command.zox;
    else if (code == 52) commandType = Command.zix;
    else if (code == 48) commandType = Command.zrx;
    else if (code == 192) commandType = Command.zry;
    else if (code == 83) commandType = Command.search;
    else if (code == 67) commandType = Command.clear_search;
    else if (code == 84) commandType = Command.toggle_search;
    else if (code == 72) commandType = Command.search_history;
    else if (code == 80) commandType = Command.previous_search;
    else if (code == 78) commandType = Command.next_search;
  }
  else {
    if (code == 61 || code == 187) {
      if (multiFnKeys) commandType = Command.ziy;
      else commandType = Command.zix;
    }
    else if (code == 189 || code == 173) {
      if (multiFnKeys) commandType = Command.zoy;
      else commandType = Command.zox;
    }
    else if (code == 48) {
      if (multiFnKeys) commandType = Command.zry;
      else commandType = Command.zrx;
    }
  }

  if (commandType == Command.none) {
    if (e.metaKey || e.altKey)
      rangeZoom = false;
    return true;
  }

  suppressdefault(e);
  if (commandType == Command.help) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    drawHelpBox();
    setKeyHandler(makeModalKeyHandler([191, 27], function(code) {
      var timelineSvg = d3.select("#timeline").select("svg");
      timelineSvg.select("g.help-box").remove();
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }
  else if (commandType == Command.search) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    drawSearchBox();
    setKeyHandler(makeModalKeyHandler([13, 27], function(code) {
      if (code == 13) {
        var re = $("input.search-box").val();
        console.log("Search Expression: " + re);
        if (re.trim() != "") {
          if (searchRegex == null) {
            searchRegex = new Array(sizeHistory);
            currentPos = -1;
            nextPos = 0;
          }
          currentPos = nextPos;
          nextPos = (nextPos + 1) % sizeHistory;
          searchRegex[currentPos] = new RegExp(re);
          searchEnabled = true;
        }
      }
      var timelineSvg = d3.select("#timeline").select("svg");
      timelineSvg.select("g.search-box").remove();
      if (searchEnabled) {
        var timelineGroup = timelineSvg.select("g#timeline");
        filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
        drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
      }
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }
  else if (commandType == Command.search_history) {
    turnOffMouseHandlers();
    makeTimelineTransparent();
    drawSearchHistoryBox();
    setKeyHandler(makeModalKeyHandler([72, 27], function(code) {
      var timelineSvg = d3.select("#timeline").select("svg");
      timelineSvg.select("g.history-box").remove();
      makeTimelineOpaque();
      setKeyHandler(defaultKeydown);
      turnOnMouseHandlers();
    }));
    return false;
  }

  showLoaderIcon();
  if (commandType == Command.zix) {
    var inc = 4.0;
    if (zoom + inc <= maxZoom)
      adjustZoom(zoom + inc, true);
  }
  else if (commandType == Command.zox) {
    var dec = 4.0;
    if (zoom - dec > 0)
      adjustZoom(zoom - dec, true);
  }
  else if (commandType == Command.zrx) {
    adjustZoom(1.0, false);
    if ($("#timeline").scrollLeft() != 0)
      $("#timeline").scrollLeft(0);
    else {
      var svg = d3.select("#timeline").select("svg");
      var timelineGroup = svg.select("g#timeline");
      filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
      drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
    }
  }
  else if (commandType == Command.ziy)
    adjustThickness(thickness * 2);
  else if (commandType == Command.zoy)
    adjustThickness(thickness / 2);
  else if (commandType == Command.zry) {
    var height = $(window).height() - margin_bottom;
    thickness = height / max_level;
    adjustThickness(thickness);
  }
  else if (commandType == Command.clear_search) {
    searchEnabled = false;
    searchRegex = null;
    var timelineGroup = timelineSvg.select("g#timeline");
    filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
    drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
  }
  else if (commandType == Command.toggle_search) {
    if (searchRegex != null) {
      searchEnabled = !searchEnabled;
      var timelineGroup = timelineSvg.select("g#timeline");
      filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
      drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
    }
  }
  else if (commandType == Command.previous_search ||
           commandType == Command.next_search) {
    if (searchEnabled) {
      var pos = commandType == Command.previous_search ?
                (currentPos - 1 + sizeHistory) % sizeHistory :
                (currentPos + 1) % sizeHistory;
      var sentinel = commandType == Command.previous_search ?
                (nextPos - 1 + sizeHistory) % sizeHistory : nextPos;
      if (pos != sentinel && searchRegex[pos] != null) {
        currentPos = pos;
        var timelineGroup = timelineSvg.select("g#timeline");
        filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
        drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
      }
    }
  }
  hideLoaderIcon();
  return false;
}


function defaultKeyUp(e) {
  if (!e) e = event;
  if (!(e.metaKey || e.altKey)) {
    rangeZoom = true;
  }
  return true;
}

function init() {
  setKeyHandler(defaultKeydown);
  $(document).on("keyup", defaultKeyUp);
}



width = $("body").width() - margin_left - margin_right - 16;
scale = width / (end - start);
init();

var timelineSvg = d3.select("#timeline").append("svg")
  .attr("width", zoom * width)
  .attr("height", height);
var loaderSvg = d3.select("#loader-icon").append("svg")
  .attr("width", "40px")
  .attr("height", "40px");
drawLoaderIcon();

var operations = {}

var timeline_loader = function(operation_data) { 
  
  operation_data.forEach(function(d) {
    operations[parseInt(d.op_id)] = d.operation;      
  });

  d3.tsv('legion_prof_data.tsv',
    function(d, i) {
        var start = +d.start;
        var end = +d.end;
        var total = end - start;
        if (total > 10) {
            return {
                id: i,
                level: d.level,
                start: start,
                end: end,
                color: d.color,
                opacity: d.opacity,
                initiation: d.initiation,
                title: d.title
            };
        }
    },
    function(data) {
      var timelineGroup = timelineSvg.append("g")
          .attr("id", "timeline")
          .attr("transform", "scale(" + zoom +", 1.0)");

      // split profiling items by which level they're on
      profilingData.items = {};
      for(var i = 0; i < data.length; i++) {
        var d = data[i];
        if (d.level in profilingData.items) {
          profilingData.items[d.level].push(d);
        } else {
          profilingData.items[d.level] = [d];
        }
      }

      $("#timeline").scrollLeft(0);
      parseURLParameters();

      filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
      drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);

      var windowCenterY = $(window).height() / 2;
      $(window).scroll(function() {
          $("#loader-icon").css("top", $(window).scrollTop() + windowCenterY);
      });

      var timer = null;
      $("#timeline").scroll(function() {
          showLoaderIcon();
          if (timer !== null) {
            clearTimeout(timer);
          }
          timer = setTimeout(function() {
            filteredData = filterOnlyVisibleData(profilingData, zoom, scale);
            drawTimeline(timelineGroup, filteredData, zoom, scale, thickness);
            hideLoaderIcon();
          }, 100);
      });
      turnOnMouseHandlers();
  });
};

d3.tsv("legion_prof_ops.tsv", 
  function(d) {
    return d;
  },
  timeline_loader // callback once we load in the ops will be to load the timeline
);

d3.tsv('legion_prof_processor.tsv',
    function(d) {
      return {
        level: d.level,
        processor: d.processor
      };
    },
    function(data) {
      profilingData.processors = data;
      drawProcessors(profilingData.processors);
    }
);

function parseURLParameters() {
  var match,
  pl     = /\+/g,  // Regex for replacing addition symbol with a space
  search = /([^&=]+)=?([^&]*)/g,
  decode = function (s) { return decodeURIComponent(s.replace(pl, " ")); },
  query  = window.location.search.substring(1);

  var urlParams = {};
  while (match = search.exec(query))
    urlParams[decode(match[1])] = decode(match[2]);

  // adjust zoom
  var zstart = start;
  if ("start" in urlParams)
    zstart = Math.max(start, parseFloat(urlParams["start"]));

  var zend = end;
  if ("end" in urlParams)
    zend = Math.min(end, parseFloat(urlParams["end"]));

  if(zstart < zend) {
    // set zoom to get: (zend - start) * zoom * scale = $("#timeline").width()
    adjustZoom($("#timeline").width() / scale / (zend - zstart), false);

    // set scrollLeft to get:  zstart * zoom * scale = scrollLeft()
    $("#timeline").scrollLeft(zstart * zoom * scale);
  }

  if ("search" in urlParams) {
    searchEnabled = true;
    searchRegex = new Array(sizeHistory);
    currentPos = 0;
    nextPos = 1;
    searchRegex[currentPos] = new RegExp(urlParams["search"]);
  }
}

hideLoaderIcon();


