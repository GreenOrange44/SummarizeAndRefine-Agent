from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
from schemas import SummarizeRequest, SummarizeResponse
from graph_backend import graph, AgentState

app = FastAPI()

# endpoints

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Summarizer Agent"}

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_blocking(request: SummarizeRequest) -> SummarizeResponse:
    try:
        # Initialize the state
        initial_state = {
            "text": request.text,
            "chunks": [],
            "chunk_summaries": [],
            "final_summary": "",
            "iteration_count": 0,
            "max_words": request.max_words
        }

        # Run the graph (invoke waits for the final result)
        # Note: We are using 'ainvoke' because our nodes are async
        result = await graph.ainvoke(initial_state)

        return SummarizeResponse(
            final_summary=result["final_summary"],
            metadata={
                "iterations": result["iteration_count"],
                "chunk_count": len(result["chunks"])
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


