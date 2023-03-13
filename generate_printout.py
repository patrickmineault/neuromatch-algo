import pystache
import pandas as pd


def generate_printout(matches):
    template = """
<!DOCTYPE html>
<html>
<head>
    <title>Table {{number}}</title>
    <style>
    header {
        background-color: black;
        color: white;
        font-size: 2em;
        text-align: center;
        padding: 10px;
        margin-top: 10px;
        height: 100px;
        align-items: center;
        display: flex;
    }
    
    body {
        font-family: Arial, sans-serif;
    }
    
    .container {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        padding-top: 30px; 
    }
    .column {
        height: 2.7in;
    }
    .column.left {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6em;
        width: 10%;
    }

    .column.left p {
        width: 200px;
        transform: rotate(-90deg);
    }

    .column.middle {
        width: 30%;
        text-align: center;
        font-size: 1.6em;
        align-items: center;

        display: flex;
        flex-direction: column;
        vertical-align: center;
        justify-content: center;
    }
    .column.middle p {
        margin: 0 0 10px 0;
    }
    .column.right {
        width: 60%;
    }
    .column.right img {
        width: 100%;
    }
    @media print {
        .top-container {
            page-break-after: always;
            page-break-inside: avoid;
        }
    }

    .spacer {
        width: 100px;
    }

    .header-text {
        font-size: 2em;
        font-weight: bold;
        flex: 1;
    }
    </style>
</head>
<body>
    {{#tables}}
    <div class="top-container">
        <header>
            <div class="spacer"></div>
		    <h1 class="header-text">Table {{table}}</h1>
		    <div class="spacer"><img height="100" width="100" src="qr-code-cosyne.png"></div>
        </header>
        {{#rounds}}
        <div class="container">
            <div class="column left"><p>Round&nbsp;{{round}}</p></div>
            <div class="column middle">
                <p>{{user_name_0}}</p>
                <p>{{user_name_1}}</p>
                <p>{{user_name_2}}</p>
                <p>{{user_name_3}}</p>
            </div>
            <div class="column right"><img src="output/keywords_{{round}}_{{table}}.png" alt=""></div>
        </div>
        <hr />
        {{/rounds}}
    </div>
    {{/tables}}
</body>
</html>
    """

    records = {"records": matches.to_dict(orient="records")}
    tables = {}
    for row in records["records"]:
        if row["table"] not in tables:
            tables[row["table"]] = {"table": row["table"], "rounds": [row]}
        else:
            tables[row["table"]]["rounds"].append(row)

    data = {"tables": list(tables.values())}
    html = pystache.render(template, data)

    with open("html/printouts.html", "w") as f:
        f.write(html)


def main():
    df_matches = pd.read_csv("data/output/matches_with_annotations.csv")

    generate_printout(df_matches)


if __name__ == "__main__":
    main()
